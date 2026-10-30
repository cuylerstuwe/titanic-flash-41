//! Finite repeat diagnostics: retain first-arm draws on GPU and copy only the
//! first differing pair to host after the second request. No model arithmetic.
use crate::cuda::{Buffer,Cuda,Module,Stream};
use anyhow::{Result,ensure};
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::path::Path;

// Six live rows: logits(F32), p/q(F64), three uniforms and three integer arrays.
const STRIDE:usize=(6*129280*4+(6+5)*129280*8+6*(3*8+3*4)+255)&!255;
const CAPACITY:usize=192; // no request in this finite screen emits more than192
struct Frame { metadata:Value, regions:Vec<(String,usize)> }
struct PeerCopyPlan { sources:Vec<(u64,usize)>, table:Buffer<u64> }
pub(crate) struct DrawTrace {
    cuda:Cuda, storage:Buffer<u8>, frames:Vec<Frame>, reference:bool,
    compared:usize, difference:Option<(usize,Frame)>, memory_before:Value,
    stride:usize, layer_trace:bool,
    byte_compare:Option<(Module,Buffer<u32>)>, comparison_frames:Vec<Frame>,
    capacity:usize, deep_trace:bool, reference_seen:usize, compact_peer:bool,
    peer_batched:bool, peer_plans:Vec<PeerCopyPlan>, peer_copy_checks:usize,
    peer_detail:bool,
}
impl DrawTrace {
    pub(crate) fn new(cuda:&Cuda,core:&[u8],compact_peer:bool)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_READOUT_BUFFERED_TRACE").as_deref()==Ok("1")
            &&std::env::var("CANADA_SCRATCH_SAMPLER_READOUT_SCREEN").as_deref()==Ok("1"),
            "buffered draws require finite readout diagnosis");
        let layer_trace=std::env::var("CANADA_SCRATCH_READOUT_LAYER_TRACE").as_deref()==Ok("1");
        let compare_bytes=std::env::var("CANADA_SCRATCH_READOUT_COMPARE_BYTES").as_deref()==Ok("1");
        let deep_trace=std::env::var("CANADA_SCRATCH_READOUT_DEEP_TRACE").as_deref()==Ok("1");
        ensure!(!deep_trace||(layer_trace&&compare_bytes),"deep repeat capture needs byte and layer recording");
        ensure!(!compact_peer||deep_trace,"compact peer capture requires deep source boundaries");
        let peer_batched=compact_peer
            &&std::env::var("CANADA_SCRATCH_READOUT_TRACE_PEER_BATCHED").as_deref()==Ok("1");
        let peer_detail=compact_peer
            &&std::env::var("CANADA_SCRATCH_READOUT_TRACE_PEER_DETAIL").as_deref()==Ok("1");
        ensure!(!peer_detail||peer_batched,"detailed peers require batched copying");
        // Trade the number of saved steps for more boundaries, not model
        // precision or the UMA reserve. The known first difference is step24;
        // requests continue normally beyond this explicitly reported window.
        let capacity=if deep_trace {72}else{CAPACITY};
        let boundary_bytes=6*5120*2+if deep_trace {2*6*5120*2+6*20480*2+6*24*4}else{0};
        // Peers need only the owned local/reduced values to distinguish a
        // changed contribution from a transport/reduction defect. Keep root's
        // existing recording unchanged and avoid large peer probability copies.
        let stride=if peer_detail {80*boundary_bytes+40*(6*(1152*2+576*2+5120*2+6*4*2+384*4+5120)+20480)}
            else if compact_peer {80*2*6*5120*2}
            else {STRIDE+if layer_trace {80*boundary_bytes}else{0}};
        let bytes=stride*(capacity+1+usize::from(compare_bytes));let memory=cuda.admission_memory()?;
        ensure!(memory.admitted_available>bytes+(8usize<<30)+(64<<20),
            "buffered draw diagnosis would consume the UMA reserve");
        let storage=cuda.allocate(bytes)?;
        let byte_compare=if compare_bytes {Some((cuda.module(core)?,cuda.allocate(1)?))}else{None};
        ensure!(cuda.admission_memory()?.admitted_available>(8usize<<30)+(64<<20),
            "buffered draw allocation consumed the UMA reserve");
        Ok(Self {cuda:cuda.clone(),storage,frames:Vec::new(),reference:true,
            compared:0,difference:None,memory_before:serde_json::to_value(memory)?,stride,layer_trace,
            byte_compare,comparison_frames:Vec::new(),capacity,deep_trace,reference_seen:0,compact_peer,
            peer_batched,peer_plans:Vec::new(),peer_copy_checks:0,peer_detail})
    }
    pub(crate) fn compact_peer(&self)->bool {self.compact_peer}
    pub(crate) fn peer_detail(&self)->bool {self.peer_detail}
    pub(crate) fn begin(&mut self,reference:bool,stream:&Stream)->Result<()> {
        stream.synchronize()?;
        self.reference=reference;
        if reference {
            self.frames.clear();self.difference=None;self.compared=0;self.comparison_frames.clear();self.reference_seen=0;
            if let Some((_,first))=&mut self.byte_compare {first.upload(&[u32::MAX])?;}
        }
        else {ensure!(!self.frames.is_empty(),"buffered comparison has no reference draws");}
        Ok(())
    }
    pub(crate) fn record(&mut self,metadata:Value,sources:&[(String,u64,usize)],stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&sources.len()==if self.peer_detail {880}else if self.compact_peer {160}
            else {9+if self.layer_trace {80*if self.deep_trace {7}else{1}}else{0}},"draw capture sources/context");
        let slot=if self.reference {
            self.reference_seen+=1;
            if self.deep_trace&&self.frames.len()==self.capacity{return Ok(());}
            ensure!(self.frames.len()<self.capacity,"buffered draw capacity exceeded");
            self.frames.len()
        }else{
            let index=self.compared;self.compared+=1;
            if self.deep_trace&&index>=self.capacity{return Ok(());}
            if self.byte_compare.is_some() {
                ensure!(index<self.capacity,"byte comparison frame capacity");self.capacity+1
            }else{
                if self.difference.is_some(){return Ok(());}
                ensure!(index<self.frames.len(),"comparison has extra draws without an earlier metadata difference");
                if self.frames[index].metadata==metadata{return Ok(());}
                self.capacity
            }
        };
        let mut offset=0;let mut regions=Vec::with_capacity(sources.len());
        for (name,source,bytes)in sources {
            if self.compact_peer&&!self.peer_detail&&!(name.ends_with("-local")||name.ends_with("-reduced")){continue;}
            ensure!(*bytes>0&&bytes%4==0&&offset+bytes<=self.stride,"draw capture extent");
            if !self.peer_batched {
                unsafe {self.cuda.copy_device(self.storage.ptr()+((slot*self.stride+offset)as u64),*source,*bytes,stream)?;}
            }
            regions.push((name.clone(),*bytes));offset+=bytes;
        }
        if self.peer_batched {
            let key:Vec<_>=sources.iter().map(|(_,p,b)|(*p,*b)).collect();
            let (plan,new)=if let Some(i)=self.peer_plans.iter().position(|p|p.sources==key){(i,false)}else{
                ensure!(self.peer_plans.len()<5,"peer copy source owners changed beyond live widths2..6");
                let mut words=Vec::with_capacity(key.len()*3);let mut cursor=0;
                for (p,b) in &key {words.extend([*p,cursor as u64,(*b/4)as u64]);cursor+=b;}
                let mut table=self.cuda.allocate(words.len())?;table.upload(&words)?;stream.join_default()?;
                self.peer_plans.push(PeerCopyPlan{sources:key,table});(self.peer_plans.len()-1,true)
            };
            let (module,_)=self.byte_compare.as_ref().expect("compact peer byte comparison");
            let mut table=self.peer_plans[plan].table.ptr();
            let mut destination=self.storage.ptr()+(slot*self.stride)as u64;
            unsafe {module.function("ca_trace_copy_regions")?.launch([sources.len()as u32,1,1],[256,1,1],0,stream,
                &mut[(&mut table as *mut u64).cast(),(&mut destination as *mut u64).cast()])?;}
            if new {
                // Once per live width, qualify the diagnostic copy against
                // driver byte copies of the same completed model buffers.
                // The extra work occurs only on first use; all arithmetic
                // remains outside this integer-only tracing path.
                ensure!(self.cuda.admission_memory()?.admitted_available>offset+(8usize<<30),
                    "peer copy check would consume the UMA reserve");
                let check=self.cuda.allocate::<u8>(offset)?;let mut flag=self.cuda.allocate::<u32>(1)?;
                flag.upload(&[u32::MAX])?;stream.join_default()?;
                let mut cursor=0;let mut control=check.ptr();
                for (_,p,b) in sources {unsafe{self.cuda.copy_device(control+cursor as u64,*p,*b,stream)?;}cursor+=b;}
                let mut selected=flag.ptr();let mut count=(offset/4)as i32;let mut ordinal=0u32;let mut forced=0i32;
                unsafe {module.function("ca_trace_first_difference")?.launch([256,1,1],[256,1,1],0,stream,
                    &mut[(&mut destination as *mut u64).cast(),(&mut control as *mut u64).cast(),
                        (&mut selected as *mut u64).cast(),(&mut count as *mut i32).cast(),
                        (&mut ordinal as *mut u32).cast(),(&mut forced as *mut i32).cast()])?;}
                stream.synchronize()?;ensure!(flag.download()?[0]==u32::MAX,"batched peer byte-copy mismatch");
                self.peer_copy_checks+=1;
            }
        }
        let frame=Frame {metadata,regions};
        if self.reference {self.frames.push(frame);}
        else if let Some((module,first))=&self.byte_compare {
            let index=self.compared-1;
            let mut forced=i32::from(self.frames.get(index).is_none_or(|r|
                r.metadata!=frame.metadata||r.regions!=frame.regions));
            let mut current=self.storage.ptr()+((self.capacity+1)*self.stride)as u64;
            let mut reference=self.storage.ptr()+(index.min(self.frames.len()-1)*self.stride)as u64;
            let mut selected=first.ptr();let mut words=(offset/4)as i32;let mut ordinal=index as u32;
            let mut saved=self.storage.ptr()+(self.capacity*self.stride)as u64;
            let grid=[((words as u32).div_ceil(256)).min(256),1,1];
            unsafe {
                module.function("ca_trace_first_difference")?.launch(grid,[256,1,1],0,stream,
                    &mut[(&mut current as *mut u64).cast(),(&mut reference as *mut u64).cast(),
                        (&mut selected as *mut u64).cast(),(&mut words as *mut i32).cast(),
                        (&mut ordinal as *mut u32).cast(),(&mut forced as *mut i32).cast()])?;
                module.function("ca_trace_copy_first")?.launch(grid,[256,1,1],0,stream,
                    &mut[(&mut current as *mut u64).cast(),(&mut saved as *mut u64).cast(),
                        (&mut selected as *mut u64).cast(),(&mut words as *mut i32).cast(),
                        (&mut ordinal as *mut u32).cast()])?;
            }
            self.comparison_frames.push(frame);
        }
        else {self.difference=Some((self.compared-1,frame));}
        Ok(())
    }
    fn write_frame(&self,frame:&Frame,slot:usize,path:&Path)->Result<()> {
        std::fs::create_dir(path)?;let mut metadata=frame.metadata.clone();let mut offset=0;
        for (name,bytes)in &frame.regions {
            let raw=unsafe {self.cuda.read_device(self.storage.ptr()+((slot*self.stride+offset)as u64),*bytes)}?;
            if name=="logits"||name=="q" {metadata[format!("{name}_sha256")]=json!(format!("{:x}",Sha256::digest(&raw)));}
            std::fs::write(path.join(format!("{name}.bin")),raw)?;offset+=bytes;
        }
        std::fs::write(path.join("metadata.json"),serde_json::to_vec_pretty(&metadata)?)?;Ok(())
    }
    pub(crate) fn finish_pair(&self,path:&Path,modes:[u8;2],stream:&Stream)->Result<Value> {
        ensure!(!self.reference,"buffered pair not compared");stream.synchronize()?;
        std::fs::create_dir(path)?;
        let selected=if let Some((_,first))=&self.byte_compare {
            let index=first.download()?[0];
            if index==u32::MAX {None}else{
                let index=index as usize;
                ensure!(index<self.frames.len()&&index<self.comparison_frames.len(),"selected byte pair extent");
                Some((index,&self.comparison_frames[index]))
            }
        }else{self.difference.as_ref().map(|(i,f)|(*i,f))};
        if let Some((index,frame))=selected {
            self.write_frame(&self.frames[index],index,&path.join(modes[0].to_string()))?;
            self.write_frame(frame,self.capacity,&path.join(modes[1].to_string()))?;
        }
        let report=json!({"reference_draws":self.reference_seen,"compared_draws":self.compared,
            "recorded_reference_draws":self.frames.len(),"capture_window":self.capacity,
            "capture_window_truncated":self.reference_seen>self.capacity||self.compared>self.capacity,
            "first_different_draw":selected.map(|x|x.0),"modes":modes,
            "captured_pair":selected.is_some(),"allocated_bytes":self.storage.bytes(),
            "compare_tensor_bytes":self.byte_compare.is_some(),
            "selected_metadata_equal":selected.map(|(i,f)|self.frames[i].metadata==f.metadata),
            "layer_trace":self.layer_trace,"deep_trace":self.deep_trace,
            "compact_peer_reductions":self.compact_peer,
            "peer_batched_copy":self.peer_batched,"peer_copy_width_checks":self.peer_copy_checks,
            "peer_detailed_boundaries":self.peer_detail,
            "memory_before":self.memory_before,"memory_after":self.cuda.admission_memory()?,
            "scope":"Asynchronous device copies of reference rejection draws with2–6rows. Optional GPU integer byte comparison selects the first changed tensor or metadata frame, otherwise metadata only. History point-mass and single-row draws are outside this capture. Host tensor reads only after the second request. Diagnostic, not serving performance."});
        std::fs::write(path.join("capture.json"),serde_json::to_vec_pretty(&report)?)?;Ok(report)
    }
}
