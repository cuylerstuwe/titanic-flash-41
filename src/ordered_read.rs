//! Bounded two-buffer read-ahead. The consumer remains on its calling thread.
use std::{io::{self, Read}, sync::mpsc::sync_channel, time::Instant};

#[derive(Default, Debug)]
pub struct Timing { pub read_seconds:f64, pub wait_seconds:f64, pub buffers:usize }

pub fn chunks<R:Read+Send,F:FnMut(usize,&[u8])->io::Result<()>>(
    input:&mut R,total:usize,chunk:usize,pipeline:bool,mut consume:F,
)->io::Result<Timing> {
    if chunk==0 || chunk>16*1024*1024 {return Err(io::Error::other("invalid read chunk"));}
    let mut timing=Timing::default();
    if total==0 {return Ok(timing);}
    let chunk=chunk.min(total);
    if !pipeline {
        let mut buffer=vec![0;chunk];timing.buffers=1;
        for offset in (0..total).step_by(chunk) {
            let n=chunk.min(total-offset);let start=Instant::now();
            input.read_exact(&mut buffer[..n])?;
            timing.read_seconds+=start.elapsed().as_secs_f64();
            consume(offset,&buffer[..n])?;
        }
        timing.wait_seconds=timing.read_seconds;
        return Ok(timing);
    }
    let (spare_tx,spare_rx)=sync_channel::<Vec<u8>>(2);
    let (ready_tx,ready_rx)=sync_channel::<io::Result<(usize,Vec<u8>,usize,f64)>>(1);
    // Exactly two buffers in flight across both channels and the consumer.
    for _ in 0..2 {spare_tx.send(vec![0;chunk]).unwrap();}
    timing.buffers=2;
    std::thread::scope(|scope| {
        let worker=scope.spawn(move|| {
            for offset in (0..total).step_by(chunk) {
                let Ok(mut buffer)=spare_rx.recv()else{return;};
                let n=chunk.min(total-offset);let start=Instant::now();
                if let Err(error)=input.read_exact(&mut buffer[..n]) {
                    let _=ready_tx.send(Err(error));return;
                }
                if ready_tx.send(Ok((offset,buffer,n,start.elapsed().as_secs_f64()))).is_err(){return;}
            }
        });
        let result=(|| {
            for expected in (0..total).step_by(chunk) {
                let start=Instant::now();
                let (offset,buffer,n,seconds)=ready_rx.recv()
                    .map_err(|_|io::Error::other("read-ahead worker stopped"))??;
                timing.wait_seconds+=start.elapsed().as_secs_f64();timing.read_seconds+=seconds;
                if offset!=expected || n!=chunk.min(total-expected) {
                    return Err(io::Error::other("read-ahead ordering/extent"));
                }
                consume(offset,&buffer[..n])?;
                // The worker can already have delivered the final chunk and
                // exited. Returning that last spare is then unnecessary.
                let _=spare_tx.send(buffer);
            }
            Ok(())
        })();
        // A consumer error must unblock either producer wait before joining.
        drop(spare_tx);drop(ready_rx);
        worker.join().map_err(|_|io::Error::other("read-ahead worker panicked"))?;
        result
    })?;
    Ok(timing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test] fn ordered_tail_and_zero() {
        for pipeline in [false,true] {for len in [0,1,15,16,17,1031] {
            let data:Vec<_>=(0..len).map(|x|(x*17)as u8).collect();let mut out=vec![];
            let timing=chunks(&mut Cursor::new(&data),len,16,pipeline,|offset,bytes|{
                assert_eq!(out.len(),offset);out.extend_from_slice(bytes);Ok(())
            }).unwrap();assert_eq!(out,data);assert!(timing.buffers<=2);
        }}
    }
    #[test] fn truncated_source_is_an_error() {
        for pipeline in [false,true] {
            let e=chunks(&mut Cursor::new(vec![1u8;31]),32,16,pipeline,|_,_|Ok(())).unwrap_err();
            assert_eq!(e.kind(),io::ErrorKind::UnexpectedEof);
        }
    }
    #[test] fn consumer_failure_joins_without_hanging() {
        for stop in [0,16,32] {
            let e=chunks(&mut Cursor::new(vec![1u8;64]),64,16,true,|offset,_|{
                if offset==stop{Err(io::Error::other("consumer failure"))}else{Ok(())}
            }).unwrap_err();assert_eq!(e.to_string(),"consumer failure");
        }
    }
    #[test] fn callback_runs_on_original_thread() {
        let id=std::thread::current().id();
        chunks(&mut Cursor::new(vec![1u8;65]),65,16,true,|_,_|{
            assert_eq!(std::thread::current().id(),id);Ok(())
        }).unwrap();
    }
    #[test] fn consumer_panic_releases_producer_before_scope_join() {
        let result=std::panic::catch_unwind(|| {
            let _=chunks(&mut Cursor::new(vec![1u8;1024]),1024,16,true,|_,_|{
                panic!("consumer panic");
            });
        });
        assert!(result.is_err());
    }
    #[test] fn reader_panic_is_a_joined_error() {
        struct PanickingReader;
        impl Read for PanickingReader {
            fn read(&mut self,_:&mut[u8])->io::Result<usize>{panic!("reader panic");}
        }
        let error=chunks(&mut PanickingReader,64,16,true,|_,_|Ok(())).unwrap_err();
        assert_eq!(error.to_string(),"read-ahead worker panicked");
    }
    #[test] fn invalid_extent_is_refused() {
        assert!(chunks(&mut Cursor::new(vec![]),1,0,true,|_,_|Ok(())).is_err());
        assert!(chunks(&mut Cursor::new(vec![]),1,16*1024*1024+1,true,|_,_|Ok(())).is_err());
    }
}
