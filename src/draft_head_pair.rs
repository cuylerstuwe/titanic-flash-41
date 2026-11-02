//! Pool only the row-independent vocabulary projection. Packing and scattering
//! preserve each owner's BF16 activations and private FP32 base logits.
use super::*;
use crate::cuda::Graph;

pub(crate) struct Pair {
    // Captured pointers must retire before their backing storage.
    graph:Option<Graph>,
    bindings:Option<[u64;6]>,
    cuda:Cuda,
    projection:Operator,
    input:Buffer<bf16>,
    output:Buffer<f32>,
    stream:usize,
}
impl DraftHead {
    pub(crate) fn can_pool_with(&self,other:&Self)->bool {
        Rc::ptr_eq(&self.canonical,&other.canonical)&&self.stream==other.stream
            &&self.quantizer_sha256==other.quantizer_sha256
            &&self.projection.entry.artifact_sha256==other.projection.entry.artifact_sha256
            && !self.gemv_active.get()&&!other.gemv_active.get()
            && !self.integer_active.get()&&!other.integer_active.get()
            && !self.lossless_active.get()&&!other.lossless_active.get()
    }
}
impl Pair {
    pub(crate) fn new(cuda:&Cuda,root:&Path,stream:&Stream)->Result<Self> {
        let projection=Operator::load(cuda,root,"draft_head_m10")?;
        ensure!(projection.entry.artifact_sha256=="236e4854586055d35fc03488c9d5c8c26ab2282c9bbf5f8d3ba9e2b686ee9709"
            &&projection.entry.arguments.iter().map(|a|(a.name.as_str(),a.kind.as_str())).collect::<Vec<_>>()
                ==[("X","*bf16"),("W8","*fp8e4nv"),("S","*u8"),("Y","*fp32")]
            &&projection.entry.grid.rows_div==32&&projection.entry.grid.x_mul==1
            &&projection.entry.grid.x_add==0&&projection.entry.grid.y==1&&projection.entry.grid.z==1,
            "paired draft head must use the qualified M10/FP32 operator");
        ensure!(cuda.owns_stream(stream)&&cuda.admission_memory()?.admitted_available
            >=8*1024usize.pow(3)+2*M*(K*2+N*4),"paired draft head reserve");
        Ok(Self{graph:None,bindings:None,cuda:cuda.clone(),projection,
            input:cuda.allocate(2*M*K)?,output:cuda.allocate(2*M*N)?,stream:stream.raw()as usize})
    }
    pub(crate) fn run(&mut self,heads:[&DraftHead;2],inputs:[&Buffer<bf16>;2],
        outputs:[&Buffer<f32>;2],stream:&Stream)->Result<()> {
        ensure!(self.cuda.owns_stream(stream)&&self.stream==stream.raw()as usize
            &&heads[0].stream==self.stream&&heads[0].can_pool_with(heads[1])
            &&inputs.iter().all(|b|self.cuda.owns_buffer(b)&&b.len()==M*K)
            &&outputs.iter().all(|b|self.cuda.owns_buffer(b)&&b.len()==M*N)
            &&inputs[0].ptr()!=inputs[1].ptr()&&outputs[0].ptr()!=outputs[1].ptr(),
            "paired head source/context/private output contract");
        let bindings=[inputs[0].ptr(),inputs[1].ptr(),outputs[0].ptr(),outputs[1].ptr(),
            heads[0].weights.ptr(),heads[0].scales.ptr()];
        if self.bindings!=Some(bindings) {
            stream.synchronize()?;self.graph=None;
            self.graph=Some(stream.capture(||self.enqueue(bindings,stream))?);
            self.bindings=Some(bindings);
        }
        unsafe{self.graph.as_ref().expect("bound paired head graph").launch(stream)}
    }
    fn enqueue(&self,p:[u64;6],stream:&Stream)->Result<()> {
        unsafe {
            for owner in 0..2 {
                self.cuda.copy_device(self.input.ptr()+(owner*M*K*2)as u64,p[owner],M*K*2,stream)?;
            }
            self.projection.launch(N as u32,&[self.input.ptr(),p[4],p[5],self.output.ptr()],stream)?;
            for owner in 0..2 {
                self.cuda.copy_device(p[2+owner],self.output.ptr()+(owner*M*N*4)as u64,M*N*4,stream)?;
            }
        }
        Ok(())
    }
}
