//! Three bounded stages: file read, ordered SHA256, and owner-thread consumption.
//! Only byte buffers cross threads; the CUDA consumer keeps its original context.
use std::{io::{self,Read},sync::mpsc::sync_channel,time::Instant};
use sha2::{Digest,Sha256};
type Piece=(usize,Vec<u8>,usize,f64);
fn closed()->io::Error {io::Error::other("ordered read/hash stage stopped")}
pub(crate) fn chunks<R:Read+Send,F:FnMut(usize,&[u8])->io::Result<()>>(
    input:&mut R,total:usize,chunk:usize,mut consume:F,
)->io::Result<(crate::ordered_read::Timing,String,f64)> {
    if chunk==0||chunk>16*1024*1024{return Err(io::Error::other("invalid hash pipeline chunk"));}
    if total==0{return Ok((Default::default(),format!("{:x}",Sha256::digest([])),0.));}
    let chunk=chunk.min(total);
    let(spare_tx,spare_rx)=sync_channel::<Vec<u8>>(3);
    let(read_tx,read_rx)=sync_channel::<io::Result<Piece>>(1);
    let(hashed_tx,hashed_rx)=sync_channel::<io::Result<Piece>>(1);
    for _ in 0..3{spare_tx.send(vec![0;chunk]).unwrap();}
    std::thread::scope(move|scope|{
        let reader=scope.spawn(move||{
            for offset in (0..total).step_by(chunk){
                let Ok(mut buffer)=spare_rx.recv()else{return;};
                let n=chunk.min(total-offset);let at=Instant::now();
                if let Err(e)=input.read_exact(&mut buffer[..n]){let _=read_tx.send(Err(e));return;}
                if read_tx.send(Ok((offset,buffer,n,at.elapsed().as_secs_f64()))).is_err(){return;}
            }
        });
        let hasher=scope.spawn(move||->io::Result<(String,f64)>{
            let mut hash=Sha256::new();let mut seconds=0.;
            for expected in (0..total).step_by(chunk){
                let piece=match read_rx.recv().map_err(|_|closed()).and_then(|v|v){
                    Ok(p)=>p,Err(e)=>{let _=hashed_tx.send(Err(e));return Err(closed());}
                };
                let(offset,buffer,n,_) = &piece;
                if *offset!=expected||*n!=chunk.min(total-expected){
                    let _=hashed_tx.send(Err(io::Error::other("read/hash ordering/extent")));return Err(closed());
                }
                let at=Instant::now();hash.update(&buffer[..*n]);seconds+=at.elapsed().as_secs_f64();
                hashed_tx.send(Ok(piece)).map_err(|_|closed())?;
            }
            Ok((format!("{:x}",hash.finalize()),seconds))
        });
        let mut timing=crate::ordered_read::Timing{buffers:3,..Default::default()};
        let result=(||{
            for expected in (0..total).step_by(chunk){
                let at=Instant::now();let(offset,buffer,n,read_seconds)=hashed_rx.recv().map_err(|_|closed())??;
                timing.wait_seconds+=at.elapsed().as_secs_f64();timing.read_seconds+=read_seconds;
                if offset!=expected||n!=chunk.min(total-expected){return Err(io::Error::other("hashed stream ordering/extent"));}
                consume(offset,&buffer[..n])?;
                let _=spare_tx.send(buffer);
            }
            Ok(())
        })();
        // These owners also drop before scope-join during consumer unwinding.
        // Disconnecting both ends unblocks either upstream producer wait.
        drop(spare_tx);drop(hashed_rx);
        let read_join=reader.join();let hash_join=hasher.join();
        read_join.map_err(|_|io::Error::other("read/hash reader panicked"))?;
        let hashed=hash_join.map_err(|_|io::Error::other("read/hash hasher panicked"))?;
        result?;let(digest,seconds)=hashed?;Ok((timing,digest,seconds))
    })
}

#[cfg(test)] mod tests {
    use super::*;use std::io::Cursor;
    #[test]fn digest_and_owner_with_tails(){
        for len in [0,1,16,17,129]{
            let bytes:Vec<u8>=(0..len).map(|n|(n*31)as u8).collect();let mut seen=vec![];
            let owner=std::thread::current().id();
            let(t,d,_)=chunks(&mut Cursor::new(&bytes),len,16,|offset,data|{
                assert_eq!(std::thread::current().id(),owner);assert_eq!(offset,seen.len());seen.extend_from_slice(data);Ok(())
            }).unwrap();assert_eq!(seen,bytes);assert_eq!(d,format!("{:x}",Sha256::digest(&bytes)));assert!(t.buffers<=3);
        }
    }
    #[test]fn failures_join_both_producers(){
        let e=chunks(&mut Cursor::new(vec![0;31]),32,16,|_,_|Ok(())).unwrap_err();assert_eq!(e.kind(),io::ErrorKind::UnexpectedEof);
        for fail in [0,16,48]{
            let e=chunks(&mut Cursor::new(vec![0;64]),64,16,|offset,_|{
                if offset==fail{Err(io::Error::other("consumer failed"))}else{Ok(())}
            }).unwrap_err();assert_eq!(e.to_string(),"consumer failed");
        }
    }
    #[test]fn panics_release_both_producers(){
        assert!(std::panic::catch_unwind(||{let _=chunks(&mut Cursor::new(vec![0;1024]),1024,16,|_,_|panic!("consumer"));}).is_err());
        struct Broken;impl Read for Broken{fn read(&mut self,_:&mut[u8])->io::Result<usize>{panic!("reader");}}
        assert_eq!(chunks(&mut Broken,64,16,|_,_|Ok(())).unwrap_err().to_string(),"read/hash reader panicked");
    }
}
