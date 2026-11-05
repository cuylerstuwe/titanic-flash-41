//! Diagnostic snapshots of private query outputs, never causal model state.
use super::*;

pub(crate) struct QueryOutputSnapshot {
    base:Buffer<f32>,hidden:Buffer<bf16>,pub(crate) position:u64,
}
impl QueryNetwork {
    pub(crate) fn snapshot_query_outputs_for_probe(&self,stream:&Stream)->Result<QueryOutputSnapshot>{
        ensure!(std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1")
            &&self.ready&&!self.poisoned&&!self.gather_base_logits&&self.cuda.owns_stream(stream),
            "finite ready local-query snapshot");
        let base=self.cuda.allocate(5*32320)?;let hidden=self.cuda.allocate(5*5120)?;
        unsafe{
            self.cuda.copy_device(base.ptr(),self.io.local_logits().ptr(),base.bytes(),stream)?;
            self.cuda.copy_device(hidden.ptr(),self.combined.ptr(),hidden.bytes(),stream)?;
        }
        stream.synchronize()?;
        Ok(QueryOutputSnapshot{base,hidden,position:self.position()?})
    }
    pub(crate) fn install_shifted_query_for_probe(&mut self,snapshot:&QueryOutputSnapshot,
        offset:usize,stream:&Stream)->Result<()> {
        ensure!(std::env::var("CANADA_SCRATCH_QUERY_REUSE_SCREEN").as_deref()==Ok("1")
            &&self.ready&&!self.poisoned&&!self.gather_base_logits&&offset<5
            &&snapshot.position.checked_add(offset as u64)==Some(self.position()?)
            &&self.cuda.owns_buffer(&snapshot.base)&&self.cuda.owns_buffer(&snapshot.hidden)
            &&self.cuda.owns_stream(stream),"finite shifted query identity");
        for row in 0..5 {
            // Only5-offset rows have a genuine old prediction. Repeating the
            // last row defines harmless discarded suffix scratch; this probe
            // measures the first proposal only and publishes no sampled token.
            let source=(row+offset).min(4);
            unsafe{
                self.cuda.copy_device(self.io.local_logits().ptr()+(row*32320*4)as u64,
                    snapshot.base.ptr()+(source*32320*4)as u64,32320*4,stream)?;
                self.cuda.copy_device(self.combined.ptr()+(row*5120*2)as u64,
                    snapshot.hidden.ptr()+(source*5120*2)as u64,5120*2,stream)?;
            }
        }
        self.markov.invalidate_results();Ok(())
    }
}
