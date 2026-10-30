//! Isolated next-round length experiment; GPU computes probability-derived
//! progress and controller arithmetic. This is not saved serving policy state.
use crate::cuda::{Buffer,Cuda,MappedHost,Module,Stream};
use anyhow::{ensure,Result};
use serde_json::{json,Value};

pub(crate) struct BlockProgress {
    cuda:Cuda,module:Module,expected:Buffer<f64>,state:Buffer<f64>,counts:Buffer<u32>,
    readout:MappedHost,enabled:bool,validate:bool,control_state:Buffer<u32>,control_ids:Buffer<i32>,
}
impl BlockProgress {
    pub(crate) fn new(cuda:&Cuda,core:&[u8])->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN").as_deref()==Ok("1"),"isolated block progress");
        Ok(Self{cuda:cuda.clone(),module:cuda.module(core)?,expected:cuda.allocate(6)?,
            state:cuda.allocate(4)?,counts:cuda.allocate(4)?,readout:cuda.mapped_host(96)?,enabled:false,
            validate:false,control_state:cuda.allocate(6)?,control_ids:cuda.allocate(6)?})
    }
    pub(crate) fn pointer(&self)->Option<u64> {self.enabled.then_some(self.expected.ptr())}
    pub(crate) fn controls(&self)->Option<(u64,u64)> {
        (self.enabled&&self.validate).then_some((self.control_state.ptr(),self.control_ids.ptr()))
    }
    pub(crate) fn reset(&mut self,enabled:bool,validate:bool,stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream),"block progress stream");
        unsafe {
            self.cuda.zero_device(self.state.ptr(),self.state.bytes(),stream)?;
            self.cuda.zero_device(self.counts.ptr(),self.counts.bytes(),stream)?;
            self.cuda.zero_device(self.expected.ptr(),self.expected.bytes(),stream)?;
        }
        stream.synchronize()?;self.enabled=enabled;self.validate=validate;Ok(())
    }
    pub(crate) fn observe(&mut self,rows:usize,seconds:f64,stream:&Stream)->Result<(usize,Value)> {
        ensure!(self.enabled&&[3,6].contains(&rows)&&seconds.is_finite()&&seconds>0.
            &&self.cuda.owns_stream(stream),"block progress committed observation");
        let mut ep=self.expected.ptr();let mut state=self.state.ptr();let mut counts=self.counts.ptr();
        let mut out=self.readout.device_ptr();let mut n=rows as i32;let mut elapsed=seconds;
        unsafe {self.module.function("ca_block_progress_update")?.launch([1,1,1],[32,1,1],0,stream,
            &mut[(&mut ep as *mut u64).cast(),(&mut state as *mut u64).cast(),
                (&mut counts as *mut u64).cast(),(&mut out as *mut u64).cast(),
                (&mut n as *mut i32).cast(),(&mut elapsed as *mut f64).cast()])?;}
        stream.synchronize()?;
        ensure!(self.readout.status_word(0)?==0,"invalid block progress observation");
        let next=self.readout.status_word(4)?as usize;ensure!([3,6].contains(&next),"block progress next rows");
        let mut values=Vec::new();
        for i in 0..10 {
            let low=self.readout.status_word(16+i*8)?as u64;
            let high=self.readout.status_word(20+i*8)?as u64;
            let value=f64::from_bits(low|(high<<32));ensure!(value.is_finite(),"block progress readout");values.push(value);
        }
        Ok((next,json!({"rows":rows,"next_rows":next,"seconds":seconds,
            "observation":self.readout.status_word(8)?,"expected":&values[..rows],
            "ema_expected":&values[6..8],"ema_seconds":&values[8..10]})))
    }
}
