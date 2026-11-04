//! Full ordered SHA256 with bounded overlap of device readback and CPU hashing.
//! The read callback stays on its owning thread; only bytes cross to the worker.
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::sync::mpsc::sync_channel;

pub(crate) struct Part {pub prefix:Vec<u8>,pub offset:usize,pub bytes:usize}

pub(crate) fn digest<F:FnMut(usize,&mut[u8])->Result<()>>(
    parts:&[Part],chunk:usize,mut read:F,
)->Result<String>{
    ensure!(chunk>0&&chunk<=16*1024*1024,"digest chunk bound");
    ensure!(parts.iter().all(|p|p.offset.checked_add(p.bytes).is_some()),"digest extent overflow");
    let (spare_tx,spare_rx)=sync_channel::<Vec<u8>>(2);
    let (ready_tx,ready_rx)=sync_channel::<(usize,usize,Vec<u8>,usize)>(1);
    for _ in 0..2{spare_tx.send(vec![0;chunk])?;}
    std::thread::scope(move|scope|{
        let worker=scope.spawn(move||->Result<String>{
            let mut hash=Sha256::new();
            for (part_index,part) in parts.iter().enumerate(){
                hash.update(&part.prefix);
                for offset in (0..part.bytes).step_by(chunk){
                    let(index,at,buffer,n)=ready_rx.recv()?;
                    ensure!(index==part_index&&at==offset&&n==chunk.min(part.bytes-offset),
                        "digest stream ordering/extent");
                    hash.update(&buffer[..n]);
                    // Returning the final spare is unnecessary once the caller
                    // has submitted every chunk and closed its receiving end.
                    let _=spare_tx.send(buffer);
                }
            }
            Ok(format!("{:x}",hash.finalize()))
        });
        let produced=(||->Result<()>{
            for(part_index,part)in parts.iter().enumerate(){
                for offset in (0..part.bytes).step_by(chunk){
                    let mut buffer=spare_rx.recv()?;
                    let n=chunk.min(part.bytes-offset);
                    read(part.offset+offset,&mut buffer[..n])?;
                    ready_tx.send((part_index,offset,buffer,n))?;
                }
            }
            Ok(())
        })();
        // Also owned by this scope on unwind: a failed read or panic cannot
        // leave the worker waiting for bytes or a spare receiver forever.
        drop(ready_tx);drop(spare_rx);
        let hashed=worker.join().map_err(|_|anyhow::anyhow!("digest worker panicked"))?;
        produced?;hashed
    })
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn ordered_headers_empty_sections_and_tails(){
        let source:Vec<_>=(0..257).map(|i|(i*13)as u8).collect();
        let parts=[Part{prefix:b"first".to_vec(),offset:193,bytes:64},
            Part{prefix:b"empty".to_vec(),offset:17,bytes:0},
            Part{prefix:b"last".to_vec(),offset:0,bytes:113}];
        let mut serial=Sha256::new();
        for p in &parts{serial.update(&p.prefix);serial.update(&source[p.offset..p.offset+p.bytes]);}
        let expected=format!("{:x}",serial.finalize());
        for chunk in [1,16,64,128]{
            let owner=std::thread::current().id();
            assert_eq!(digest(&parts,chunk,|offset,out|{
                assert_eq!(std::thread::current().id(),owner);
                out.copy_from_slice(&source[offset..offset+out.len()]);Ok(())
            }).unwrap(),expected);
        }
    }
    #[test]fn failed_read_returns_no_digest_and_joins(){
        for stop in [0,16,48]{
            let error=digest(&[Part{prefix:vec![],offset:0,bytes:64}],16,|offset,out|{
                ensure!(offset!=stop,"injected read failure");out.fill(5);Ok(())
            }).unwrap_err();assert_eq!(error.to_string(),"injected read failure");
        }
    }
    #[test]fn panicking_read_unblocks_worker(){
        assert!(std::panic::catch_unwind(||{
            let _=digest(&[Part{prefix:vec![],offset:0,bytes:1024}],16,|_,_|panic!("read panic"));
        }).is_err());
    }
    #[test]fn empty_input_and_bad_extent(){
        assert_eq!(digest(&[],16,|_,_|panic!("empty read")).unwrap(),format!("{:x}",Sha256::digest([])));
        assert!(digest(&[],0,|_,_|Ok(())).is_err());
        assert!(digest(&[Part{prefix:vec![],offset:usize::MAX,bytes:1}],16,|_,_|Ok(())).is_err());
    }
}
