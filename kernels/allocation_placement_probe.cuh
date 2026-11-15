// Read-only diagnostic of existing weight allocations. Integer XOR makes the
// readback stable for arbitrary checkpoint bit patterns, including float NaNs.
extern "C" __global__ void ca_allocation_probe_read(const uint4* input,
 unsigned long long vectors, uint4* output) {
    unsigned long long t=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned long long step=gridDim.x*blockDim.x;
    uint4 sum=make_uint4(0,0,0,0);
    for(unsigned long long i=t;i<vectors;i+=step) {
        unsigned a,b,c,d;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
            :"=r"(a),"=r"(b),"=r"(c),"=r"(d):"l"(input+i):"memory");
        sum.x^=a;sum.y^=b;sum.z^=c;sum.w^=d;
    }
    output[t]=sum;
}
// Block-contiguous streams over the same bytes: block b sweeps the b-th slice in order, as a tile-streaming GEMM
// does (many independent sequential streams). On GB10 this pattern is the one that slows down when the range mixes
// the two halves of physical memory; the single wavefront above only sees mixing finer than its own width.
extern "C" __global__ void ca_allocation_probe_streams(const uint4* input,
 unsigned long long vectors, uint4* output) {
    unsigned long long slice=vectors/gridDim.x;
    const uint4* base=input+slice*blockIdx.x;
    uint4 sum=make_uint4(0,0,0,0);
    for(unsigned long long i=threadIdx.x;i<slice;i+=blockDim.x) {
        unsigned a,b,c,d;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
            :"=r"(a),"=r"(b),"=r"(c),"=r"(d):"l"(base+i):"memory");
        sum.x^=a;sum.y^=b;sum.z^=c;sum.w^=d;
    }
    output[blockIdx.x*blockDim.x+threadIdx.x]=sum;
}
