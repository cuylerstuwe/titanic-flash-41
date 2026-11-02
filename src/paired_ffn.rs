//! Native paired FFN execution with private routing and nonlinear state.
//! Routed experts and optional shared-up projections pool independent rows.
use super::*;

pub(crate) struct Paired {
    cuda: Cuda,
    workspace: Workspace,
    input: Buffer<bf16>,
    ids: Buffer<i32>,
    routing: Buffer<f32>,
    sum_input: Buffer<bf16>,
    sum_output: Buffer<bf16>,
    shared_up: Option<crate::fp8_linear::PairedSharedUp>,
    qkv: Option<crate::fp8_linear::PairedQkv>,
}
impl Paired {
    pub(crate) fn new(cuda:&Cuda,core:&[u8],kernels:&Path)->Result<Self> {
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_FFN_SCREEN").as_deref()==Ok("1")
            ||crate::config::paired_serving_requested(),"paired FFN requires explicit admission");
        let mut input=cuda.allocate::<bf16>(16*5120)?;input.zero()?;
        let mut ids=cuda.allocate::<i32>(16*6)?;ids.zero()?;
        let mut routing=cuda.allocate::<f32>(16*6)?;routing.zero()?;
        let shared_up=match std::env::var("CANADA_SCRATCH_PAIRED_SHARED_UP") {
            Ok(value)=>{ensure!(value=="0"||value=="1","paired shared-up flag");
                if value=="1"{Some(crate::fp8_linear::PairedSharedUp::new(cuda,core,kernels)?)}else{None}},
            Err(std::env::VarError::NotPresent)=>None,Err(error)=>return Err(error.into()),
        };
        let qkv=match std::env::var("CANADA_SCRATCH_PAIRED_QKV") {
            Ok(value)=>{ensure!(value=="0"||value=="1","paired QKV flag");
                if value=="1"{Some(crate::fp8_linear::PairedQkv::new(cuda,core,kernels)?)}else{None}},
            Err(std::env::VarError::NotPresent)=>None,Err(error)=>return Err(error.into()),
        };
        Ok(Self{cuda:cuda.clone(),workspace:Workspace::new_live16(cuda,kernels)?,input,ids,routing,
            sum_input:cuda.allocate(12*5120)?,sum_output:cuda.allocate(12*5120)?,shared_up,qkv})
    }
    pub(crate) fn pooled_shared_up(&self)->bool {self.shared_up.is_some()}
    pub(crate) fn qkv(&self)->Option<&crate::fp8_linear::PairedQkv> {self.qkv.as_ref()}
    /// One fixed-rank BF16 reduction over the concatenated row domains. Views
    /// retain these allocations; captured graphs never depend on a stack view.
    pub(crate) fn reduce(&self,a:&Buffer<bf16>,b:&Buffer<bf16>,a_live:usize,b_live:usize,comm:&crate::roce::Roce,
        stream:&Stream)->Result<[Buffer<bf16>;2]> {
        ensure!(std::env::var("CANADA_SCRATCH_PAIRED_REDUCE_SCREEN").as_deref()==Ok("1")
            &&a.len()==6*5120&&b.len()==6*5120&&self.cuda.owns_buffer(a)&&self.cuda.owns_buffer(b)
            &&(1..=6).contains(&a_live)&&(1..=6).contains(&b_live)
            &&self.cuda.owns_stream(stream),"finite paired reduction domain");
        unsafe {
            self.cuda.copy_device(self.sum_input.ptr(),a.ptr(),a_live*5120*2,stream)?;
            self.cuda.copy_device(self.sum_input.ptr()+(a_live*5120*2)as u64,b.ptr(),b_live*5120*2,stream)?;
            let input=self.sum_input.view(0,(a_live+b_live)*5120)?;
            let mut output=self.sum_output.view(0,(a_live+b_live)*5120)?;
            // No per-owner prefix scope may truncate the combined domain.
            let _rows=comm.native_row_scope(false,false,a_live+b_live,12,stream)?;
            comm.sum_bf16(&input,&mut output,stream)?;
            Ok([self.sum_output.view(0,a_live*5120)?,self.sum_output.view(a_live*5120,b_live*5120)?])
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(&self,a:&LayerFfn,a_input:&Buffer<bf16>,
        a_quant:Option<(&Buffer<u8>,&Buffer<u8>)>,a_tokens:&TokenRows,
        b:&LayerFfn,b_input:&Buffer<bf16>,b_quant:Option<(&Buffer<u8>,&Buffer<u8>)>,
        b_tokens:&TokenRows,stream:&Stream)->Result<()> {
        self.run_with_shared_pool(a,a_input,a_quant,a_tokens,b,b_input,b_quant,b_tokens,true,stream)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_with_shared_pool(&self,a:&LayerFfn,a_input:&Buffer<bf16>,
        a_quant:Option<(&Buffer<u8>,&Buffer<u8>)>,a_tokens:&TokenRows,
        b:&LayerFfn,b_input:&Buffer<bf16>,b_quant:Option<(&Buffer<u8>,&Buffer<u8>)>,
        b_tokens:&TokenRows,use_shared_pool:bool,stream:&Stream)->Result<()> {
        let a_live=a_tokens.active_rows();let b_live=b_tokens.active_rows();
        ensure!(a.rows==6&&b.rows==6&&(1..=6).contains(&a_live)&&(1..=6).contains(&b_live)
            &&a_input.len()==6*5120&&b_input.len()==6*5120
            &&a.expert.same_layer_weights(&b.expert)&&!a.use_row_transport&&!b.use_row_transport
            &&self.cuda.owns_buffer(a_input)&&self.cuda.owns_buffer(b_input)&&self.cuda.owns_stream(stream),
            "paired FFN needs two live prefixes of one canonical M6 layer");
        for ffn in [a,b] {
            if let Some(branch)=&ffn.shared_branch {branch.input_ready.record(stream)?;}
        }
        a.router.run(a_input,a_tokens,stream)?;
        b.router.run(b_input,b_tokens,stream)?;
        // All six copies are graph-capturable, on the same ordered stream.
        // The unused suffix is outside the live launch. Owners may have
        // different live lengths; no padding from A becomes a live row of B.
        for (offset,live,ffn,input) in [(0,a_live,a,a_input),(a_live,b_live,b,b_input)] {
            unsafe {
                self.cuda.copy_device(self.input.ptr()+(offset*5120*2)as u64,input.ptr(),live*5120*2,stream)?;
                self.cuda.copy_device(self.ids.ptr()+(offset*6*4)as u64,ffn.router.ids().ptr(),live*6*4,stream)?;
                self.cuda.copy_device(self.routing.ptr()+(offset*6*4)as u64,ffn.router.routing().ptr(),live*6*4,stream)?;
            }
        }
        self.workspace.run_live(&a.expert,&self.input,&self.ids,&self.routing,a_live+b_live,stream)?;
        if let Some(pool)=self.shared_up.as_ref().filter(|_|use_shared_pool) {
            let(aq,asc)=a_quant.context("paired shared-up needs prequantized normalized input A")?;
            let(bq,bsc)=b_quant.context("paired shared-up needs prequantized normalized input B")?;
            let shared=match (&a.shared_branch,&b.shared_branch) {
                (Some(x),Some(y))=>{ensure!(Rc::ptr_eq(&x.stream,&y.stream),"paired shared branches need the same ordered auxiliary channel");
                    x.input_ready.wait(&x.stream)?;Some(x)},
                (None,None)=>None,
                _=>anyhow::bail!("paired shared branches have different overlap policies"),
            };
            let side=shared.map_or(stream,|branch|branch.stream.as_ref());
            pool.run(&a.shared_up,aq,asc,a_live,&b.shared_up,bq,bsc,b_live,side)?;
            a.finish_shared(side)?;b.finish_shared(side)?;
            if let Some(branch)=shared {branch.output_ready.record(side)?;branch.output_ready.wait(stream)?;}
        }else{for (ffn,input,quantized) in [(a,a_input,a_quant),(b,b_input,b_quant)] {
            if let Some(branch)=&ffn.shared_branch {
                branch.input_ready.wait(&branch.stream)?;
                ffn.run_shared_quantized_input(input,quantized,&branch.stream)?;
                branch.output_ready.record(&branch.stream)?;
                branch.output_ready.wait(stream)?;
            }else{ffn.run_shared_quantized_input(input,quantized,stream)?;}
        }}
        for (offset,live,ffn) in [(0,a_live,a),(a_live,b_live,b)] {
            ensure!(unsafe{(ffn.native.add)(live as i32,
                (self.workspace.output().ptr()+(offset*5120*2)as u64)as *const c_void,
                ffn.shared_down.output().ptr()as *const c_void,
                ffn.output.ptr()as *mut c_void,stream.raw())}==0,
                "paired FFN shared addition failed");
        }
        Ok(())
    }
}
