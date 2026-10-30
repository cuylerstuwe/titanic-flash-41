//! Persistent CPU control for one four-rank native worker group.
//!
//! Rank zero supplies commands. Every rank acknowledges the same sequence and
//! can compare a small committed-state record before publishing output. No GPU
//! polling or model math runs here. The caller sends idle heartbeats while no
//! request is ready; a broken connection permanently retires this group.
use anyhow::{Context, Result, ensure};
use std::{
    io::{Read, Write},
    net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream},
    time::{Duration, Instant},
    sync::{Arc, atomic::{AtomicU64, Ordering}},
};

const MAGIC: &[u8; 8] = b"CACTRL01";
const COMMAND: u8 = 1;
const AGREEMENT: u8 = 2;
const ACK: u8 = 3;
const COMMAND_COMMIT: u8 = 4;
const AGREEMENT_NEXT: u8 = 5;
const OPTIONAL_ADMISSION: u8 = 6;
// Four bytes/token plus44bytes/image (at least four tokens), and framing.
pub(crate) const MAX_COMMAND: usize = 16 * crate::config::MODEL_CONTEXT + 4096;
const MAX_AGREEMENT: usize = 4096;
const MAX_NEXT_COMMAND: usize = 16;

pub(crate) struct ControlChannel {
    rank: usize,
    // Rank zero owns peers1..3 in rank order; a follower owns only rank zero.
    peers: Vec<TcpStream>,
    sequence: u64,
    poisoned: bool,
    // Only completed all-rank exchanges advance the serving liveness clock.
    progress: Option<Arc<AtomicU64>>,
}

fn configure(stream: &TcpStream, timeout: Duration) -> Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(())
}

fn write_frame(stream: &mut impl Write, kind: u8, sequence: u64, body: &[u8]) -> Result<()> {
    ensure!(body.len() <= MAX_COMMAND, "control frame exceeds bound");
    // One write avoids a second tiny TCP packet on the per-step path.
    let mut frame = Vec::with_capacity(21 + body.len());
    frame.extend_from_slice(MAGIC);
    frame.push(kind);
    frame.extend_from_slice(&sequence.to_le_bytes());
    frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
    frame.extend_from_slice(body);
    stream.write_all(&frame)?;
    Ok(())
}

fn read_frame(stream: &mut impl Read, kind: u8, sequence: u64, bound: usize) -> Result<Vec<u8>> {
    let mut header = [0; 21];
    stream.read_exact(&mut header)?;
    ensure!(
        &header[..8] == MAGIC
            && header[8] == kind
            && u64::from_le_bytes(header[9..17].try_into().unwrap()) == sequence,
        "control frame identity or operation sequence differs"
    );
    let bytes = u32::from_le_bytes(header[17..21].try_into().unwrap()) as usize;
    ensure!(
        bytes <= bound && bytes <= MAX_COMMAND,
        "control frame exceeds receive bound"
    );
    let mut body = vec![0; bytes];
    stream.read_exact(&mut body)?;
    Ok(body)
}

impl ControlChannel {
    pub(crate) fn observe_progress(&mut self, counter: Arc<AtomicU64>) {
        counter.store(self.sequence, Ordering::Release);
        self.progress = Some(counter);
    }
    pub(crate) fn retire(&mut self) {
        self.poisoned = true;
        for peer in &self.peers {
            let _ = peer.shutdown(Shutdown::Both);
        }
    }
    pub(crate) fn connect(
        rank: usize,
        endpoint: SocketAddr,
        rank_ips: [IpAddr; 4],
        ticket: &[u8],
        identity: &[u8; 32],
        timeout: Duration,
    ) -> Result<Self> {
        ensure!(
            rank < 4
                && ticket.len() == 32
                && endpoint.ip() == rank_ips[0]
                && (Duration::from_millis(100)..=Duration::from_secs(600)).contains(&timeout),
            "native control group configuration"
        );
        let deadline = Instant::now() + timeout;
        let peers = if rank == 0 {
            let listener = TcpListener::bind(endpoint)?;
            listener.set_nonblocking(true)?;
            let mut joined: [Option<TcpStream>; 3] = [None, None, None];
            while joined.iter().any(Option::is_none) {
                ensure!(Instant::now() < deadline, "native control join timeout");
                match listener.accept() {
                    Ok((mut stream, address)) => {
                        // A malformed or absent participant invalidates this
                        // group; never resume its GPU state under a new peer.
                        configure(
                            &stream,
                            deadline
                                .saturating_duration_since(Instant::now())
                                .max(Duration::from_millis(1)),
                        )?;
                        let mut hello = [0; 73];
                        stream.read_exact(&mut hello)?;
                        let peer = hello[8] as usize;
                        ensure!(
                            &hello[..8] == MAGIC
                                && (1..4).contains(&peer)
                                && hello[9..41] == *ticket
                                && hello[41..] == *identity
                                && address.ip() == rank_ips[peer]
                                && joined[peer - 1].is_none(),
                            "native control peer identity mismatch"
                        );
                        joined[peer - 1] = Some(stream);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            let mut peers: Vec<_> = joined.into_iter().map(Option::unwrap).collect();
            for peer in &mut peers {
                configure(peer, timeout)?;
                write_frame(peer, ACK, 0, identity)?;
            }
            peers
        } else {
            let mut stream = loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                ensure!(!remaining.is_zero(), "native control connection timeout");
                match TcpStream::connect_timeout(&endpoint, remaining.min(Duration::from_secs(1))) {
                    Ok(stream) => break stream,
                    Err(_) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    Err(error) => return Err(error).context("native control connection"),
                }
            };
            configure(&stream, timeout)?;
            let mut hello = Vec::with_capacity(73);
            hello.extend_from_slice(MAGIC);
            hello.push(rank as u8);
            hello.extend_from_slice(ticket);
            hello.extend_from_slice(identity);
            stream.write_all(&hello)?;
            ensure!(
                read_frame(&mut stream, ACK, 0, 32)? == identity,
                "native control group identity mismatch"
            );
            vec![stream]
        };
        Ok(Self {
            rank,
            peers,
            sequence: 1,
            poisoned: false,
            progress: None,
        })
    }

    fn operation<T>(&mut self, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        ensure!(
            !self.poisoned && self.sequence < u64::MAX,
            "native control group is retired"
        );
        self.poisoned = true;
        match run(self) {
            Ok(value) => {
                self.sequence += 1;
                self.poisoned = false;
                if let Some(progress) = &self.progress {
                    progress.store(self.sequence, Ordering::Release);
                }
                Ok(value)
            }
            Err(error) => {
                // Wake participants immediately instead of waiting for their
                // receive timeout. The model owner must also retire on error.
                self.retire();
                Err(error)
            }
        }
    }

    /// Exactly one authority supplies an opaque bounded command. The caller
    /// validates its typed semantics before GPU work. Followers cannot propose
    /// an independent cancellation, seed, request, or scheduling decision.
    pub(crate) fn command(&mut self, leader: Option<&[u8]>) -> Result<Vec<u8>> {
        self.operation(|this| {
            if this.rank == 0 {
                let body =
                    leader.ok_or_else(|| anyhow::anyhow!("rank zero omitted its command"))?;
                ensure!(
                    !body.is_empty() && body.len() <= MAX_COMMAND,
                    "native command size"
                );
                for peer in &mut this.peers {
                    write_frame(peer, COMMAND, this.sequence, body)?;
                }
                for peer in &mut this.peers {
                    ensure!(
                        read_frame(peer, ACK, this.sequence, 1)? == [1],
                        "native command was not acknowledged"
                    );
                }
                for peer in &mut this.peers {
                    write_frame(peer, COMMAND_COMMIT, this.sequence, &[1])?;
                }
                Ok(body.to_vec())
            } else {
                ensure!(leader.is_none(), "only rank zero may supply a command");
                let body = read_frame(&mut this.peers[0], COMMAND, this.sequence, MAX_COMMAND)?;
                ensure!(!body.is_empty(), "empty native command");
                write_frame(&mut this.peers[0], ACK, this.sequence, &[1])?;
                ensure!(
                    read_frame(&mut this.peers[0], COMMAND_COMMIT, this.sequence, 1)? == [1],
                    "native command did not reach the complete worker group"
                );
                Ok(body)
            }
        })
    }

    /// Compare committed integer state or its identity before client emission.
    /// This does not compare floating-point tensors or require reproducible
    /// trajectories between different engine executions.
    pub(crate) fn agree(&mut self, local: &[u8]) -> Result<()> {
        self.operation(|this| {
            ensure!(
                !local.is_empty() && local.len() <= MAX_AGREEMENT,
                "native agreement size"
            );
            if this.rank == 0 {
                let mut equal = true;
                for peer in &mut this.peers {
                    equal &= read_frame(peer, AGREEMENT, this.sequence, MAX_AGREEMENT)? == local;
                }
                for peer in &mut this.peers {
                    write_frame(peer, ACK, this.sequence, &[u8::from(equal)])?;
                }
                ensure!(equal, "native ranks disagree on committed state");
            } else {
                write_frame(&mut this.peers[0], AGREEMENT, this.sequence, local)?;
                ensure!(
                    read_frame(&mut this.peers[0], ACK, this.sequence, 1)? == [1],
                    "native ranks disagree on committed state"
                );
            }
            Ok(())
        })
    }

    /// Optional memory work proceeds only when every rank can admit it.
    /// A normal refusal is a shared false result, not a poisoned worker group.
    pub(crate) fn all_admit(&mut self, local:bool)->Result<bool> {
        self.operation(|this| {
            if this.rank==0 {
                let mut admitted=local;
                for peer in &mut this.peers {
                    let vote=read_frame(peer,OPTIONAL_ADMISSION,this.sequence,1)?;
                    ensure!(vote.len()==1 && vote[0]<=1,"invalid optional admission vote");
                    admitted &= vote[0]==1;
                }
                for peer in &mut this.peers {write_frame(peer,ACK,this.sequence,&[u8::from(admitted)])?;}
                Ok(admitted)
            }else{
                write_frame(&mut this.peers[0],OPTIONAL_ADMISSION,this.sequence,&[u8::from(local)])?;
                let result=read_frame(&mut this.peers[0],ACK,this.sequence,1)?;
                ensure!(result.len()==1 && result[0]<=1,"invalid optional admission result");
                Ok(result[0]==1)
            }
        })
    }

    /// After every rank reports the same committed frontier, rank zero can
    /// authorize the next bounded command in the agreement acknowledgment.
    /// The caller must not publish this frontier until this returns. This
    /// saves a separate command handshake; it does not skip state agreement.
    /// A partially delivered acknowledgment or later peer failure still
    /// retires the group; no subsequent frontier may be published without
    /// another complete agreement. Begin/idle/shutdown use the normal command
    /// handshake, and the caller restricts these commands to step/finish.
    pub(crate) fn agree_and_next(&mut self, local: &[u8], leader: Option<&[u8]>) -> Result<Vec<u8>> {
        self.operation(|this| {
            ensure!(!local.is_empty() && local.len() <= MAX_AGREEMENT,
                "native progress agreement size");
            if this.rank == 0 {
                let command = leader.ok_or_else(|| anyhow::anyhow!("rank zero omitted next command"))?;
                ensure!(!command.is_empty() && command.len() <= MAX_NEXT_COMMAND,
                    "native next command size");
                let mut equal = true;
                for peer in &mut this.peers {
                    equal &= read_frame(peer, AGREEMENT_NEXT, this.sequence, MAX_AGREEMENT)? == local;
                }
                let mut acknowledgment = vec![u8::from(equal)];
                if equal { acknowledgment.extend_from_slice(command); }
                for peer in &mut this.peers {
                    write_frame(peer, ACK, this.sequence, &acknowledgment)?;
                }
                ensure!(equal, "native ranks disagree on committed state");
                Ok(command.to_vec())
            } else {
                ensure!(leader.is_none(), "only rank zero may authorize next command");
                write_frame(&mut this.peers[0], AGREEMENT_NEXT, this.sequence, local)?;
                let acknowledgment = read_frame(&mut this.peers[0], ACK, this.sequence, MAX_NEXT_COMMAND + 1)?;
                ensure!(acknowledgment.first() == Some(&1) && acknowledgment.len() > 1,
                    "native progress was not agreed and authorized");
                Ok(acknowledgment[1..].to_vec())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn frames_reject_wrong_order_truncation_and_oversized_allocation() {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, COMMAND, 42, &[0, 255, 7]).unwrap();
        assert_eq!(
            read_frame(&mut Cursor::new(&bytes), COMMAND, 42, 3).unwrap(),
            [0, 255, 7]
        );
        assert!(read_frame(&mut Cursor::new(&bytes), COMMAND, 43, 3).is_err());
        assert!(read_frame(&mut Cursor::new(&bytes), AGREEMENT, 42, 3).is_err());
        assert!(read_frame(&mut Cursor::new(&bytes[..23]), COMMAND, 42, 3).is_err());
        assert!(read_frame(&mut Cursor::new(&bytes), COMMAND, 42, 2).is_err());
        bytes[17..21].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(read_frame(&mut Cursor::new(&bytes), COMMAND, 42, MAX_COMMAND).is_err());
    }

    #[test]
    fn four_rank_commands_and_divergence_retire_the_group() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|rank| {
                    scope.spawn(move || {
                        let mut channel = ControlChannel::connect(
                            rank,
                            address,
                            [address.ip(); 4],
                            &[17; 32],
                            &[29; 32],
                            Duration::from_secs(5),
                        )
                        .unwrap();
                        let progress = Arc::new(AtomicU64::new(0));
                        channel.observe_progress(progress.clone());
                        assert_eq!(progress.load(Ordering::Acquire), 1);
                        assert!(channel.all_admit(true).unwrap());
                        assert_eq!(progress.load(Ordering::Acquire), 2);
                        for refusing_rank in 0..4 {assert!(!channel.all_admit(rank!=refusing_rank).unwrap());}
                        assert!(channel.all_admit(true).unwrap());
                        for step in 0..10u64 {
                            let command = step.to_le_bytes();
                            assert_eq!(
                                channel
                                    .command((rank == 0).then_some(command.as_slice()))
                                    .unwrap(),
                                command
                            );
                            channel.agree(&command).unwrap();
                        }
                        let before_failure = progress.load(Ordering::Acquire);
                        assert!(channel.agree(&[u8::from(rank == 2)]).is_err());
                        assert!(channel.command((rank == 0).then_some(&[1])).is_err());
                        assert_eq!(progress.load(Ordering::Acquire), before_failure);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn disconnect_never_becomes_successful_end_of_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|rank| {
                    scope.spawn(move || {
                        let mut channel = ControlChannel::connect(
                            rank,
                            address,
                            [address.ip(); 4],
                            &[17; 32],
                            &[29; 32],
                            Duration::from_secs(5),
                        )
                        .unwrap();
                        channel.agree(&[1]).unwrap();
                        if rank == 2 {
                            return;
                        }
                        assert!(channel.command((rank == 0).then_some(&[1])).is_err());
                        assert!(channel.agree(&[1]).is_err());
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn progress_authorizes_identical_next_commands_and_rejects_divergence() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4).map(|rank| scope.spawn(move || {
                let mut channel = ControlChannel::connect(rank, address, [address.ip(); 4],
                    &[17; 32], &[29; 32], Duration::from_secs(5)).unwrap();
                channel.command((rank == 0).then_some(&[1])).unwrap();
                for step in 0..16u64 {
                    let next: &[u8] = if step == 15 { &[3, 2] } else { &[2] };
                    assert_eq!(channel.agree_and_next(&step.to_le_bytes(),
                        (rank == 0).then_some(next)).unwrap(), next);
                }
                // A completed request returns to the full command protocol.
                channel.agree(&[7]).unwrap();
                assert_eq!(channel.command((rank == 0).then_some(&[1, 9])).unwrap(), [1, 9]);
                assert!(channel.agree_and_next(&[u8::from(rank == 2)],
                    (rank == 0).then_some(&[2])).is_err());
                assert!(channel.agree(&[1]).is_err());
                assert!(channel.command((rank == 0).then_some(&[1])).is_err());
            })).collect();
            for handle in handles { handle.join().unwrap(); }
        });
    }

    #[test]
    fn progress_peer_disconnect_prevents_publication_of_the_next_frontier() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4).map(|rank| scope.spawn(move || {
                let mut channel = ControlChannel::connect(rank, address, [address.ip(); 4],
                    &[17; 32], &[29; 32], Duration::from_secs(5)).unwrap();
                channel.agree_and_next(&[1], (rank == 0).then_some(&[2])).unwrap();
                if rank == 2 { return; }
                assert!(channel.agree_and_next(&[2], (rank == 0).then_some(&[2])).is_err());
                assert!(channel.agree(&[3]).is_err());
            })).collect();
            for handle in handles { handle.join().unwrap(); }
        });
    }
}
