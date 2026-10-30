//! Small private-fabric rendezvous for isolated rank qualification.
use crate::collective::{Nccl, UniqueId};
use anyhow::{Context, Result, ensure};
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    time::{Duration, Instant},
};

pub fn exchange_blob(
    rank: usize,
    endpoint: SocketAddr,
    nonce: &[u8],
    local: &[u8],
) -> Result<Vec<u8>> {
    exchange_blob_with_timeout(rank, endpoint, nonce, local, Duration::from_secs(30))
}

/// Full-model ranks can finish immutable weight loading at different times.
/// Keep this wait explicit and bounded by the outer stationary watchdog; a
/// primitive probe's short default is inappropriate after tens of GiB of I/O.
pub fn exchange_blob_with_timeout(
    rank: usize,
    endpoint: SocketAddr,
    nonce: &[u8],
    local: &[u8],
    timeout: Duration,
) -> Result<Vec<u8>> {
    ensure!(
        rank < 4
            && nonce.len() == 32
            && local.len() > 0
            && local.len() <= 65536
            && (Duration::from_secs(1)..=Duration::from_secs(600)).contains(&timeout),
        "invalid endpoint exchange"
    );
    if rank == 0 {
        let listener = TcpListener::bind(endpoint)?;
        listener.set_nonblocking(true)?;
        let deadline = Instant::now() + timeout;
        let mut all = vec![0u8; 4 * local.len()];
        all[..local.len()].copy_from_slice(local);
        let mut peers = Vec::new();
        let mut joined = [true, false, false, false];
        while peers.len() < 3 {
            ensure!(Instant::now() < deadline, "endpoint exchange timeout");
            match listener.accept() {
                Ok((mut stream, peer)) => {
                    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                    let mut h = [0u8; 40];
                    stream.read_exact(&mut h)?;
                    let r = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
                    let n = u32::from_le_bytes(h[36..].try_into().unwrap()) as usize;
                    ensure!(
                        r > 0 && r < 4 && !joined[r] && h[4..36] == *nonce && n == local.len(),
                        "mismatched endpoint exchange"
                    );
                    ensure!(
                        peer.ip()
                            == std::net::IpAddr::V4(std::net::Ipv4Addr::new(
                                192,
                                168,
                                200,
                                12 + r as u8
                            )),
                        "unexpected endpoint transport"
                    );
                    stream.read_exact(&mut all[r * n..(r + 1) * n])?;
                    joined[r] = true;
                    peers.push(stream);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                Err(e) => return Err(e.into()),
            }
        }
        for p in &mut peers {
            p.write_all(&all)?;
        }
        Ok(all)
    } else {
        let deadline = Instant::now() + timeout;
        let mut s = loop {
            match TcpStream::connect_timeout(&endpoint, Duration::from_secs(2)) {
                Ok(s) => break s,
                Err(e) => {
                    if Instant::now() > deadline {
                        return Err(e.into());
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        };
        // An early peer can connect before the final peer finishes loading.
        // Its response wait must cover that same admitted loading interval.
        s.set_read_timeout(Some(timeout))?;
        s.set_write_timeout(Some(Duration::from_secs(5)))?;
        s.write_all(&(rank as u32).to_le_bytes())?;
        s.write_all(nonce)?;
        s.write_all(&(local.len() as u32).to_le_bytes())?;
        s.write_all(local)?;
        let mut all = vec![0; 4 * local.len()];
        s.read_exact(&mut all)?;
        Ok(all)
    }
}

pub fn join(api: &Nccl, rank: usize, endpoint: SocketAddr, nonce: &[u8]) -> Result<UniqueId> {
    ensure!(
        rank < 4 && nonce.len() == 32,
        "rendezvous rank/ticket shape"
    );
    if rank == 0 {
        let listener = TcpListener::bind(endpoint)?;
        listener.set_nonblocking(true)?;
        let id = api.unique_id()?;
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut joined = [true, false, false, false];
        let mut count = 0;
        while count < 3 {
            ensure!(Instant::now() < deadline, "rank rendezvous timed out");
            match listener.accept() {
                Ok((mut stream, peer)) => {
                    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                    let mut header = [0u8; 36];
                    stream.read_exact(&mut header)?;
                    let r = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
                    ensure!(
                        r > 0 && r < 4 && !joined[r] && header[4..] == *nonce,
                        "invalid or duplicate rank rendezvous"
                    );
                    let expected = std::net::Ipv4Addr::new(192, 168, 200, 12 + r as u8);
                    ensure!(
                        peer.ip() == std::net::IpAddr::V4(expected),
                        "rank connected over an unexpected transport"
                    );
                    stream.write_all(&id.bytes)?;
                    joined[r] = true;
                    count += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(id)
    } else {
        let deadline = Instant::now() + Duration::from_secs(25);
        let mut stream = loop {
            match TcpStream::connect_timeout(&endpoint, Duration::from_secs(2)) {
                Ok(s) => break s,
                Err(e) => {
                    if Instant::now() >= deadline {
                        return Err(e).context("connect rank rendezvous");
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        };
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        stream.write_all(&(rank as u32).to_le_bytes())?;
        stream.write_all(nonce)?;
        let mut id = UniqueId { bytes: [0; 128] };
        stream.read_exact(&mut id.bytes)?;
        Ok(id)
    }
}
