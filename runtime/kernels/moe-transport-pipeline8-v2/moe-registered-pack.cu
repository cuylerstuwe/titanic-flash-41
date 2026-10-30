// Original mapped-memory output experiment. The included source retains the
// qualified quantization formulas and its Apache-2.0 b12x attribution. Change
// only output organization: stage one encoded row in shared memory, then issue
// coalesced uint4 stores instead of small scattered scale/correction writes.
#include "moe-dual-quant-codec-fast.cu"

template<bool Slots=false>
__global__ void fused_pack_registered(const __nv_bfloat16* input,uint8_t* packed,
                                      int rows,const unsigned* epoch=nullptr,int rank=0,int local_rows=0) {
    constexpr int k=5120,groups=160,stride=5920;
    if constexpr(Slots)packed+=size_t(((*epoch&1u)*4)+rank)*local_rows*stride;
    __shared__ __align__(16) uint8_t encoded[stride];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
        for(int col=warp;col<groups;col+=blockDim.x/32) {
            const float x=__bfloat162float(input[row*k+col*32+lane]);
            float amax=fabsf(x);
            for(int shift=16;shift;shift/=2)
                amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,shift));
            const uint8_t sf=quant_scale(amax);
            const uint8_t q=__nv_fp8_e4m3(x*inverse_scale(sf)).__x;
            unsigned correction=0;
            if(sf<105) {
                const uint8_t shared_sf=quant_scale(fmaxf(amax,1e-4f));
                const uint8_t shared=__nv_fp8_e4m3(x*inverse_scale(shared_sf)).__x;
                float scaled;
                correction=__ballot_sync(0xffffffffu,base_shared(q,sf,scaled)!=shared);
            }
            encoded[col*32+lane]=q;
            if(lane==0) {
                encoded[k+col]=sf;
                reinterpret_cast<unsigned*>(encoded+k+groups)[col]=correction;
            }
        }
        __syncthreads();
        for(int vector=threadIdx.x;vector<stride/16;vector+=blockDim.x)
            reinterpret_cast<uint4*>(packed+size_t(row)*stride)[vector]=
                reinterpret_cast<const uint4*>(encoded)[vector];
        __syncthreads();
    }
}
extern "C" int ca_pack_registered(const void* input,void* packed,int rows,int k,cudaStream_t stream) {
    if(rows<=0||k!=5120||reinterpret_cast<uintptr_t>(packed)%16)return -1;
    fused_pack_registered<false><<<192,256,0,stream>>>(static_cast<const __nv_bfloat16*>(input),
        static_cast<uint8_t*>(packed),rows);
    return int(cudaGetLastError());
}

template<bool Partial=false,bool CopyPacked=false,bool Slots=false>
__global__ void decode_registered(const uint8_t* packed,uint8_t* shared,
                                  uint8_t* row_scales,uint8_t* mma_scales,int rows,
                                  int local_rows=0,int part_rows=0,int part=0,uint8_t* packed_copy=nullptr,
                                  const unsigned* epoch=nullptr) {
    constexpr int k=5120,groups=160,stride=5920;
    if constexpr(Slots)packed+=size_t(*epoch&1u)*4*local_rows*stride;
    __shared__ __align__(16) uint8_t encoded[stride];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int logical_row=blockIdx.x;logical_row<rows;logical_row+=gridDim.x) {
        const int row=Partial?(logical_row/part_rows)*local_rows+part*part_rows+logical_row%part_rows:logical_row;
        for(int vector=threadIdx.x;vector<stride/16;vector+=blockDim.x) {
            const uint4 value=reinterpret_cast<const uint4*>(packed+size_t(row)*stride)[vector];
            reinterpret_cast<uint4*>(encoded)[vector]=value;
            if constexpr(CopyPacked)reinterpret_cast<uint4*>(packed_copy+size_t(row)*stride)[vector]=value;
        }
        __syncthreads();
        for(int col=warp;col<groups;col+=blockDim.x/32) {
            const uint8_t q=encoded[col*32+lane],sf=encoded[k+col];
            uint8_t base=q;
            if(sf<105) {
                const unsigned correction=reinterpret_cast<const unsigned*>(encoded+k+groups)[col];
                float scaled;base=base_shared(q,sf,scaled);
                if((correction>>lane)&1u)base+=fabsf(scaled)>fabsf(f8(base))?1:-1;
            }
            encoded[col*32+lane]=base;
            if(lane==0)encoded[k+col]=max(int(sf),105);
        }
        __syncthreads();
        for(int vector=threadIdx.x;vector<k/16;vector+=blockDim.x)
            reinterpret_cast<uint4*>(shared+size_t(row)*k)[vector]=
                reinterpret_cast<const uint4*>(encoded)[vector];
        if(threadIdx.x<groups/16)
            reinterpret_cast<uint4*>(row_scales+size_t(row)*groups)[threadIdx.x]=
                reinterpret_cast<const uint4*>(encoded+k)[threadIdx.x];
        if(threadIdx.x<groups/4) {
            const int offset=(row%32)*16+((row/32)%4)*4+(row/128)*(40*512)+threadIdx.x*512;
            *reinterpret_cast<unsigned*>(mma_scales+offset)=
                reinterpret_cast<const unsigned*>(encoded+k)[threadIdx.x];
        }
        __syncthreads();
    }
}
extern "C" int ca_decode_registered(const void* packed,void* shared,void* row_scales,
                                    void* mma_scales,int rows,int k,cudaStream_t stream) {
    if(rows<=0||rows%128||k!=5120||reinterpret_cast<uintptr_t>(packed)%16)return -1;
    decode_registered<false><<<192,256,0,stream>>>(static_cast<const uint8_t*>(packed),
        static_cast<uint8_t*>(shared),static_cast<uint8_t*>(row_scales),static_cast<uint8_t*>(mma_scales),rows);
    return int(cudaGetLastError());
}

// Stream-capturable production-shaped entry points. Slot choice is a GPU
// control read, so arbitrary eager/capture/replay order never freezes parity.
extern "C" int ca_pack_registered_slot(const void* input,void* packed,const unsigned* epoch,
    int local_rows,int parts,int part,int rank,cudaStream_t stream) {
    if(!epoch||local_rows!=2048||parts!=8||part<0||part>=parts||rank<0||rank>=4||
        reinterpret_cast<uintptr_t>(packed)%16)return -1;
    const int rows=local_rows/parts;
    fused_pack_registered<true><<<192,256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(input)+size_t(part)*rows*5120,
        static_cast<uint8_t*>(packed)+size_t(part)*rows*5920,rows,epoch,rank,local_rows);
    return int(cudaGetLastError());
}
extern "C" int ca_decode_registered_slot(const void* packed,void* shared,void* row_scales,
    void* mma_scales,void* packed_copy,const unsigned* epoch,int local_rows,int parts,int part,cudaStream_t stream) {
    if(!epoch||!packed_copy||local_rows!=2048||parts!=8||part<0||part>=parts||
        reinterpret_cast<uintptr_t>(packed)%16||reinterpret_cast<uintptr_t>(packed_copy)%16)return -1;
    const int rows=local_rows/parts;
    decode_registered<true,true,true><<<192,256,0,stream>>>(static_cast<const uint8_t*>(packed),
        static_cast<uint8_t*>(shared),static_cast<uint8_t*>(row_scales),static_cast<uint8_t*>(mma_scales),
        4*rows,local_rows,rows,part,static_cast<uint8_t*>(packed_copy),epoch);
    return int(cudaGetLastError());
}
extern "C" int ca_decode_registered_part(const void* packed,void* shared,void* row_scales,
    void* mma_scales,int local_rows,int parts,int part,void* packed_copy,cudaStream_t stream) {
    if((parts!=2&&parts!=4&&parts!=8)||part<0||part>=parts||local_rows<=0||
        local_rows%(128*parts)||reinterpret_cast<uintptr_t>(packed)%16||reinterpret_cast<uintptr_t>(packed_copy)%16)return -1;
    const int part_rows=local_rows/parts;
    if(packed_copy)decode_registered<true,true><<<192,256,0,stream>>>(static_cast<const uint8_t*>(packed),
        static_cast<uint8_t*>(shared),static_cast<uint8_t*>(row_scales),static_cast<uint8_t*>(mma_scales),
        4*part_rows,local_rows,part_rows,part,static_cast<uint8_t*>(packed_copy));
    else decode_registered<true,false><<<192,256,0,stream>>>(static_cast<const uint8_t*>(packed),
        static_cast<uint8_t*>(shared),static_cast<uint8_t*>(row_scales),static_cast<uint8_t*>(mma_scales),
        4*part_rows,local_rows,part_rows,part);
    return int(cudaGetLastError());
}
