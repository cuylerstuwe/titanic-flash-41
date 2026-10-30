//! Experimental precision-preserving TP4 reduction: exchange each destination's
//! BF16 quarter, sum four original operands locally in FP32, round once, and
//! optionally gather the completed quarters. Payload volume equals a four-rank
//! ring reduce-scatter/all-reduce; actual latency needs hardware measurement.
use crate::{collective::Communicator,cuda::{Buffer,Cuda,DeviceElement,Module,Stream}};
use anyhow::{Result,ensure};
pub(crate) struct Exchange {cuda:Cuda,module:Module,received:Buffer<u16>,summed:Buffer<u16>}
impl Exchange {
    pub(crate) fn new(cuda:&Cuda,core:&[u8],elements:usize)->Result<Self>{
        ensure!(elements>0&&elements%4==0,"exchange scratch extent");
        Ok(Self{cuda:cuda.clone(),module:cuda.module(core)?,received:cuda.allocate(elements)?,summed:cuda.allocate(elements/4)?})
    }
    pub(crate) unsafe fn run<T:DeviceElement>(&self,nccl:&Communicator,input:&Buffer<T>,
        output:&mut Buffer<T>,count:usize,scatter:bool,stream:&Stream)->Result<()> {
        let n=count/4;
        ensure!(std::mem::size_of::<T>()==2&&count>0&&count%4==0&&count<=input.len()
            &&count<=self.received.len()&&self.cuda.owns_buffer(input)&&self.cuda.owns_buffer(output)
            &&self.cuda.owns_stream(stream)&&if scatter{count==output.len()*4}else{input.len()==output.len()},
            "exchange reduction geometry/owner");
        let mut received=unsafe{self.received.view(0,count)}?;
        unsafe{nccl.exchange_bf16_quarters(input,&mut received,count,stream)}?;
        let(mut source,mut destination,mut size)=(received.ptr(),if scatter{output.ptr()}else{self.summed.ptr()},n as i32);
        let mut args=[(&mut source as *mut u64).cast(),(&mut destination as *mut u64).cast(),(&mut size as *mut i32).cast()];
        unsafe{self.module.function("ca_sum_four_bf16_quarters")?.launch(
            [(n as u32).div_ceil(256).min(128),1,1],[256,1,1],0,stream,&mut args)}?;
        if !scatter {
            let sum=unsafe{self.summed.view(0,n)}?;
            unsafe{nccl.gather_bytes_prefix(&sum,output,stream)}?;
            if count<output.len(){unsafe{self.cuda.zero_device(output.ptr()+(count*2)as u64,(output.len()-count)*2,stream)}?;}
        }
        Ok(())
    }
}
