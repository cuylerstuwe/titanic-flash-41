// GPU primitives for Titanic Flash 4.1.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "state_exchange.cuh"
#include "state_image_copy.cuh"
#include "paired_dense.cuh"

// Ordered metadata for a bounded accepted draft prefix. The caller validates
// context/ring bounds; inactive lanes never name writable KV locations.
extern "C" __global__ void ca_kv_ring_metadata(int32_t* positions,int32_t* locations,
 uint32_t rows,uint32_t live,uint32_t start,uint32_t base,uint32_t ring) {
 const uint32_t row=threadIdx.x;
 if(row<rows) {
  positions[row]=row<live?int32_t(start+row):0;
  locations[row]=row<live?int32_t(base+(start+row)%ring):-1;
 }
}

// Independent implementation of NVIDIA's PTX cache-hint primitive. Rust
// validates immutable allocation bounds/alignment and retains source ownership.
// 16KiB pieces keep each 32-bit size aligned; this kernel writes no model data.
extern "C" __global__ void ca_l2_hint(const uint64_t* segments, uint32_t count) {
  constexpr uint64_t chunk = 16384;
  for (uint32_t s = blockIdx.x; s < count; s += gridDim.x) {
    uint64_t address = segments[s * 2], bytes = segments[s * 2 + 1];
    for (uint64_t off = uint64_t(threadIdx.x) * chunk; off < bytes;
         off += uint64_t(blockDim.x) * chunk) {
      uint32_t size = uint32_t(bytes - off < chunk ? bytes - off : chunk);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                   :: "l"(address + off), "r"(size) : "memory");
    }
  }
}
// s543: larger bulk requests helped the narrow input projection, but not QB.
// Same immutable byte ranges; the Rust window fixes geometry before capture.
extern "C" __global__ void ca_l2_hint_sized(const uint64_t* segments, uint32_t count, uint32_t chunk) {
  for (uint32_t s = blockIdx.x; s < count; s += gridDim.x) {
    uint64_t address = segments[s * 2], bytes = segments[s * 2 + 1];
    for (uint64_t off = uint64_t(threadIdx.x) * chunk; off < bytes;
         off += uint64_t(blockDim.x) * chunk) {
      uint32_t size = uint32_t(bytes - off < chunk ? bytes - off : chunk);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                   :: "l"(address + off), "r"(size) : "memory");
    }
  }
}
#include <math.h>
#include "collective_probe.cuh"

// Diagnostic only: one launch snapshots independently owned layer boundaries.
// The host retains every source and the immutable descriptor table until the
// ordered stream completes. Copy raw words without numerical conversion.
extern "C" __global__ void ca_trace_copy_regions(const uint64_t* table,uint8_t* destination) {
 const uint64_t* region=table+blockIdx.x*3;
 const uint32_t* source=reinterpret_cast<const uint32_t*>(region[0]);
 uint32_t* output=reinterpret_cast<uint32_t*>(destination+region[1]);
 for(uint64_t i=threadIdx.x;i<region[2];i+=blockDim.x)output[i]=source[i];
}

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float round_bf(float x) { return bf(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float logistic(float x) { return 1.f/(1.f+expf(-x)); }

// Bounded ordering diagnostic only: make a legacy-stream producer observably
// unfinished when a nonblocking consumer is launched. No model arithmetic.
extern "C" __global__ void ca_probe_delayed_word(uint32_t* word,uint32_t value,uint64_t cycles){
 if(threadIdx.x==0){uint64_t begin=clock64();while(clock64()-begin<cycles){};word[0]=value;}
}
extern "C" __global__ void ca_probe_read_word(const uint32_t* word,uint32_t* out){
 if(threadIdx.x==0)out[0]=word[0];
}

// DSpark's canonical target tap: BF16 mean of HC=4 at the attention input of
// layers37,38,39, concatenated by feature. Inputs are completed outputs36..38.
// No head/pre weighting, projection, or final-target normalization belongs here.
extern "C" __global__ void ca_dspark_target_features(
 const __nv_bfloat16* r36,const __nv_bfloat16* r37,const __nv_bfloat16* r38,
 __nv_bfloat16* output,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*15360)return;
 int row=i/15360,slot=(i/5120)%3,h=i%5120;
 const __nv_bfloat16* src=slot==0?r36:(slot==1?r37:r38);
 int base=row*20480+h;
 float sum=bf(src[base]);sum+=bf(src[base+5120]);
 sum+=bf(src[base+10240]);sum+=bf(src[base+15360]);
 output[i]=__float2bfloat16_rn(sum*0.25f);
}

// Vocabulary-parallel embedding: exactly one rank owns each admitted token.
// Copy packed BF16 words, leaving the real reduction to the TP transport.
extern "C" __global__ void ca_vocab_embedding(const int64_t* ids,const uint16_t* weight,
 uint16_t* out,int rows,int lower,int vocabulary_rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*5120)return;
 int row=i/5120,h=i%5120;int64_t token=ids[row]-lower;
 out[i]=(token>=0&&token<vocabulary_rows)?weight[token*5120+h]:0;
}
extern "C" __global__ void ca_expand_hc_embedding(const uint16_t* embedding,uint16_t* out,int rows,
 const uint64_t* image_rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*20480)return;
 int row=i/20480,h=i%5120;
 // Apply canonical image bytes after vocabulary reduction. Image values do
 // not take part in a TP sum or acquire another floating-point boundary.
 const uint16_t* image=reinterpret_cast<const uint16_t*>(image_rows[row]);
 out[i]=image?image[h]:embedding[row*5120+h];
}
extern "C" __global__ void ca_bf16_to_f32(const __nv_bfloat16* input,float* output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=__bfloat162float(input[i]);
}
extern "C" __global__ void ca_f32_to_bf16(const float* input,__nv_bfloat16* output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=__float2bfloat16_rn(input[i]);
}
extern "C" __global__ void ca_sum_four_bf16_quarters(const __nv_bfloat16* x,__nv_bfloat16* y,int n){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
  float sum=__bfloat162float(x[i]);
  sum=__fadd_rn(sum,__bfloat162float(x[n+i]));
  sum=__fadd_rn(sum,__bfloat162float(x[2*n+i]));
  sum=__fadd_rn(sum,__bfloat162float(x[3*n+i]));
  y[i]=__float2bfloat16_rn(sum);
 }
}
// Prefill emits one next-token head row. The selected live row is device
// metadata so capture never freezes a previous append's row number.
extern "C" __global__ void ca_select_last_head_row(const uint16_t* input,
 const int32_t* live,uint16_t* output,int rows){
 int h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=5120)return;
 int n=*live;output[h]=(n>=1&&n<=rows)?input[(n-1)*5120+h]:0;
}
// Each CTA owns its tracker word. Unused rows remain bitwise positive zero;
// unknown state (including any full-logit writer) requires one complete clear.
extern "C" __global__ void ca_replace_last_head_logits(const float* input,
    const int* live, int* previous, float* output, int rows) {
    const int h=blockIdx.x*blockDim.x+threadIdx.x;
    const int old=previous[blockIdx.x], now=*live;
    if(h<129280) {
        if(old>=1 && old<=rows) output[(old-1)*129280+h]=0.0f;
        else for(int r=0;r<rows;++r) output[r*129280+h]=0.0f;
        if(now>=1 && now<=rows) output[(now-1)*129280+h]=input[h];
    }
    __syncthreads();
    if(threadIdx.x==0) previous[blockIdx.x]=(now>=1&&now<=rows)?now:0;
}
extern "C" __global__ void ca_publish_last_head_logits(const float* input,
 const int32_t* live,float* output,int rows){
 int h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=129280)return;
 int n=*live;if(n>=1&&n<=rows)output[(n-1)*129280+h]=input[h];
}

// Lossless native FP8/E8M0 -> BF16 WO-A operand reconstruction. One scratch
// weight can be reused by all layers on one ordered stream; no CPU dequantizer.
extern "C" __global__ void ca_woa_decode_weight(const uint8_t* weight,const uint8_t* scales,__nv_bfloat16* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2048*4096)return;
 int row=i/4096,col=i%4096;uint8_t e=scales[(row/32)*128+col/32];
 float scale=e==0?__int_as_float(0x00400000):(e==255?__int_as_float(0x7fc00000):__int_as_float((uint32_t)e<<23));
 float x=__half2float(__nv_cvt_fp8_to_halfraw(weight[i],__NV_E4M3));
 out[i]=__float2bfloat16_rn(x*scale);
}
extern "C" __global__ void ca_woa_token_major(const uint16_t* grouped,uint16_t* out,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*2048)return;
 int row=i/2048,g=(i%2048)/1024,col=i%1024;
 out[i]=grouped[(g*rows+row)*1024+col];
}
// Keep the physical group stride; never expose untouched cuBLAS padding to a
// downstream projection or quantizer when only a prefix was computed.
extern "C" __global__ void ca_woa_token_major_bounded(const uint16_t* grouped,uint16_t* out,int rows,int active){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*2048)return;
 int row=i/2048,g=(i%2048)/1024,col=i%1024;
 out[i]=row<active?grouped[(g*rows+row)*1024+col]:0;
}

// Independent deterministic greedy target sampling. Each pass keeps the
// lowest vocabulary ID on exact ties, including signed zero. Negative infinity
// is a valid mask; NaN, positive infinity and an entirely masked row fail shut.
struct CaGreedyBest { float value; int32_t id; };
__device__ __forceinline__ CaGreedyBest ca_greedy_better(CaGreedyBest a,CaGreedyBest b){
 return (b.value>a.value||(b.value==a.value&&b.id<a.id))?b:a;
}
__device__ __forceinline__ CaGreedyBest ca_greedy_block(CaGreedyBest best){
 __shared__ float values[8];__shared__ int32_t ids[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int d=16;d;d>>=1){
  CaGreedyBest other={__shfl_down_sync(0xffffffff,best.value,d),__shfl_down_sync(0xffffffff,best.id,d)};
  best=ca_greedy_better(best,other);
 }
 if(lane==0){values[warp]=best.value;ids[warp]=best.id;}__syncthreads();
 best=(warp==0&&lane<8)?CaGreedyBest{values[lane],ids[lane]}:CaGreedyBest{-__int_as_float(0x7f800000),2147483647};
 if(warp==0)for(int d=16;d;d>>=1){
  CaGreedyBest other={__shfl_down_sync(0xffffffff,best.value,d),__shfl_down_sync(0xffffffff,best.id,d)};
  best=ca_greedy_better(best,other);
 }
 return best;
}
extern "C" __global__ void ca_greedy_partials(const float* logits,float* values,int32_t* ids,uint32_t* errors){
 int row=blockIdx.y,tile=blockIdx.x;
 CaGreedyBest best={-__int_as_float(0x7f800000),2147483647};
 for(int offset=threadIdx.x;offset<1024;offset+=256){
  int id=tile*1024+offset;if(id>=129280)continue;
  float x=logits[(size_t)row*129280+id];
  if(isnan(x)||x==__int_as_float(0x7f800000)){atomicOr(errors+row,1u);continue;}
  best=ca_greedy_better(best,CaGreedyBest{x,id});
 }
 best=ca_greedy_block(best);
 if(threadIdx.x==0){values[row*127+tile]=best.value;ids[row*127+tile]=best.id;}
}
extern "C" __global__ void ca_greedy_finish(const float* values,const int32_t* ids,uint32_t* errors,int32_t* output){
 int row=blockIdx.x,t=threadIdx.x;
 CaGreedyBest best=t<127?CaGreedyBest{values[row*127+t],ids[row*127+t]}:CaGreedyBest{-__int_as_float(0x7f800000),2147483647};
 best=ca_greedy_block(best);
 if(t==0){
  if(best.value==-__int_as_float(0x7f800000))errors[row]|=2u;
  output[row]=errors[row]?-1:best.id;
 }
}

#include "categorical.cuh"
#include "rejection.cuh"
#include "block_rejection.cuh"
#include "acceptance_expectation.cuh"
#include "sparse_index.cuh"
#include "grouped_sparse_index.cuh"

// Optional locality/order experiment: preserve the chosen top512 set exactly,
// sort absolute indices ascending, and keep -1 padding after all valid entries.
// One block owns a row; every compare/exchange reads before either side writes.
extern "C" __global__ void ca_sort_selected512(int32_t* indices) {
  __shared__ int32_t row[512];
  int t=threadIdx.x;int32_t x=indices[(size_t)blockIdx.x*512+t];
  row[t]=x<0?2147483647:x;__syncthreads();
  for(int k=2;k<=512;k<<=1)for(int j=k>>1;j>0;j>>=1) {
    int32_t a=row[t],b=row[t^j];
    bool ascending=(t&k)==0;
    int32_t next=((t&j)==0)==ascending?min(a,b):max(a,b);
    __syncthreads();row[t]=next;__syncthreads();
  }
  indices[(size_t)blockIdx.x*512+t]=row[t]==2147483647?-1:row[t];
}

// Exact integer-order alternative after native ragged topk-v2 only. That
// producer already emits ascending offset+i and -1 padding for lengths<=512.
// Warp exchanges retain the same bitonic network while avoiding CTA barriers
// for distances below32. GPU metadata is reread on every graph replay.
extern "C" __global__ void ca_sort_selected512_native(int32_t* indices, const int32_t* lengths) {
  if(lengths[blockIdx.x] <= 512)return;
  __shared__ int32_t row[512];
  int t=threadIdx.x;int32_t x=indices[(size_t)blockIdx.x*512+t];
  x=x<0?2147483647:x;
  for(int k=2;k<=512;k<<=1)for(int j=k>>1;j>0;j>>=1) {
    int32_t b;
    if(j>=32){row[t]=x;__syncthreads();b=row[t^j];__syncthreads();}
    else b=__shfl_xor_sync(0xffffffff,x,j);
    bool ascending=(t&k)==0;
    x=((t&j)==0)==ascending?min(x,b):max(x,b);
  }
  indices[(size_t)blockIdx.x*512+t]=x==2147483647?-1:x;
}

// Index pages use separate payload and exponent regions, not 68-byte AoS rows.
// Integer slots are admitted by Rust; this is a raw GPU byte gather only.
extern "C" __global__ void ca_gather_index_k(const uint8_t* cache,const int32_t* slots,
 uint32_t* payload,uint32_t* scales,int rows){
 int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
 if(row>=rows)return;
 int slot=slots[row];const uint8_t* page=cache+(int64_t)(slot/64)*4352;
 if(lane<16)payload[(int64_t)row*16+lane]=((const uint32_t*)(page+(slot%64)*64))[lane];
 if(lane==0)scales[row]=((const uint32_t*)(page+4096))[slot%64];
}
// Copy only the prefix that any live query can address, rounded to the MQA
// tile boundary. The bound is device metadata so graph replay stays current.
// Integer bytes only; no index score or canonical key is changed.
extern "C" __global__ void ca_gather_index_k_bounded(const uint8_t* cache,const int32_t* slots,
 uint32_t* payload,uint32_t* scales,int rows,const int32_t* ends,int queries){
 __shared__ int warp_ends[4];
 int lane=threadIdx.x%32,warp=threadIdx.x/32,last=0;
 for(int q=threadIdx.x;q<queries;q+=blockDim.x)last=max(last,ends[q]);
 for(int d=16;d;d>>=1)last=max(last,__shfl_down_sync(0xffffffff,last,d));
 if(lane==0)warp_ends[warp]=last;
 __syncthreads();
 last=min(rows,((max(max(warp_ends[0],warp_ends[1]),max(warp_ends[2],warp_ends[3]))+127)/128)*128);
 for(int row=blockIdx.x*4+warp;row<last;row+=gridDim.x*4){
  int slot=slots[row];const uint8_t* page=cache+(int64_t)(slot/64)*4352;
  if(lane<16)payload[(int64_t)row*16+lane]=((const uint32_t*)(page+(slot%64)*64))[lane];
  if(lane==0)scales[row]=((const uint32_t*)(page+4096))[slot%64];
 }
}
extern "C" __global__ void ca_map_index_candidates(const int32_t* selected,const int32_t* slots,
 int32_t* output,const int32_t* lengths,int32_t* output_lengths,uint32_t* error,int count,int keys,int capacity,int width){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 if(i%width==0){int n=lengths[i/width];if(n<0||n>keys)atomicOr(error,4u);
  output_lengths[i/width]=max(0,min(n,min(keys,width)));}
 int id=selected[i];if(id==-1){output[i]=-1;return;}
 if(id<0||id>=keys){atomicOr(error,1u);output[i]=-1;return;}
 int slot=slots[id];if(slot<0||slot>=capacity){atomicOr(error,2u);output[i]=-1;return;}
 output[i]=slot;
}

extern "C" __global__ void ca_expert_compact_words(const uint32_t* source,uint32_t* out,int n,int k,int group){
    int i=blockIdx.x*blockDim.x+threadIdx.x,columns=k/8;
    if(i>=n*columns)return;
    int row=i/columns,column=i%columns,rg=row/group,r=row%group,nt=r/128;
    int nr=min(128,group-nt*128),kt=column/16,c=column%16,r32=r%32;
    int dst=rg*group*columns+nt*128*columns+kt*nr*16+
      (((c/4*(nr/32)+(r%128)/32)*8+r32%8)*4+c%4)*4+r32/8;
    out[dst]=source[i];
}
extern "C" __global__ void ca_expert_compact_scales(const uint8_t* source,uint8_t* out,int n,int k,int group){
    int i=blockIdx.x*blockDim.x+threadIdx.x,columns=k/32;
    if(i>=n*columns)return;
    int row=i/columns,column=i%columns,rg=row/group,r=row%group,nt=r/128;
    int nr=min(128,group-nt*128),kt=column/4,width=min(4,columns-kt*4);
    int dst=rg*group*columns+nt*128*columns+kt*nr*4+(r%128)*width+column%4;
    out[dst]=source[i];
}

#include "expert_scale_codes.cuh"

extern "C" __global__ void ca_block_scales_mma(const uint8_t* source,uint8_t* out,int n,int k,int padded_k){
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int n_tiles=(n+127)/128,k_tiles=padded_k/128;
    if(index>=n_tiles*128*(padded_k/32))return;
    int t=index,kg=t%4;t/=4;int rh=t%4;t/=4;int rl=t%32;t/=32;
    int kt=t%k_tiles,nt=t/k_tiles;
    int row=nt*128+rh*32+rl,group=kt*4+kg;
    out[index]=(row<n&&group<k/32)?source[(row/32)*(k/32)+group]:127;
}
extern "C" __global__ void ca_pad_fp8_columns(const uint8_t* source,uint8_t* out,int n,int k,int padded_k){
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=n*padded_k)return;int row=index/padded_k,column=index%padded_k;
    out[index]=column<k?source[row*k+column]:0;
}

// Fixed pairwise warp/CTA reduction: row count never changes a row's ordering.
__device__ __forceinline__ float block_sum(float x) {
    __shared__ float partial[32];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int d=16;d;d>>=1) x+=__shfl_down_sync(0xffffffff,x,d);
    __syncthreads(); // Previous reduction's readers must finish before reuse.
    if(lane==0)partial[warp]=x;
    __syncthreads();
    x=threadIdx.x<(blockDim.x+31)/32?partial[lane]:0.f;
    if(warp==0) {for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);}
    if(threadIdx.x==0)partial[0]=x;
    __syncthreads();return partial[0];
}

extern "C" __global__ void ca_rms_bf16(const __nv_bfloat16* x,const __nv_bfloat16* w,
    __nv_bfloat16* out,int width,float eps) {
    int r=blockIdx.x;float s=0.f;
    for(int i=threadIdx.x;i<width;i+=blockDim.x){float v=bf(x[r*width+i]);s+=v*v;}
    float inv=rsqrtf(block_sum(s)/width+eps);
    for(int i=threadIdx.x;i<width;i+=blockDim.x)out[r*width+i]=__float2bfloat16_rn((bf(x[r*width+i])*inv)*bf(w[i]));
}

extern "C" __global__ void ca_hc_combine(const __nv_bfloat16* x,const float* pre,
    __nv_bfloat16* out,int width,int has_pre) {
    int r=blockIdx.x;
    for(int i=threadIdx.x;i<width;i+=blockDim.x){
        float v=bf(x[(r*4)*width+i]);
        if(has_pre){v*=pre[r*4];for(int h=1;h<4;h++)v+=bf(x[(r*4+h)*width+i])*pre[r*4+h];}
        out[r*width+i]=__float2bfloat16_rn(v);
    }
}

// Final HC boundary follows mhc.py::_hc_combine_kernel: sequential FP32 FMA
// over four streams, BF16 store, then the separately qualified CuTe RMSNorm.
// BF16->FP32 conversion is exact; no full FP32 residual temporary is needed.
extern "C" __global__ void ca_final_hc_combine(const __nv_bfloat16* x,
    const float* pre,__nv_bfloat16* out) {
    int row=blockIdx.x;
    for(int i=threadIdx.x;i<5120;i+=blockDim.x) {
        float acc=0.f;
        #pragma unroll
        for(int h=0;h<4;h++)acc=fmaf(pre[row*4+h],bf(x[(row*4+h)*5120+i]),acc);
        out[row*5120+i]=__float2bfloat16_rn(acc);
    }
}

// Last-only prompt head: preserve the same four FMA/BF16 boundary while
// avoiding preparation of rows whose logits will never be requested.
extern "C" __global__ void ca_final_hc_combine_last(const __nv_bfloat16* x,
    const float* pre,const int* live,__nv_bfloat16* out,int rows) {
    const int n=*live;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<5120;i+=gridDim.x*blockDim.x) {
        float acc=0.f;
        if(n>=1&&n<=rows) {
            const int row=n-1;
            #pragma unroll
            for(int h=0;h<4;h++)acc=fmaf(pre[row*4+h],bf(x[(row*4+h)*5120+i]),acc);
        }
        out[i]=__float2bfloat16_rn(acc);
    }
}

extern "C" __global__ void ca_hc_post(const __nv_bfloat16* x,const __nv_bfloat16* residual,
    const float* post,const float* comb,__nv_bfloat16* out,int width) {
    int r=blockIdx.x,h=blockIdx.y;
    for(int i=threadIdx.x;i<width;i+=blockDim.x){
        float v=bf(x[r*width+i])*post[r*4+h];
        for(int k=0;k<4;k++)v+=bf(residual[(r*4+k)*width+i])*comb[r*16+k*4+h];
        out[(r*4+h)*width+i]=__float2bfloat16_rn(v);
    }
}

extern "C" __global__ void ca_hc_mix(const __nv_bfloat16* x,const float* weight,float* mixes,
    int width,float eps) {
    int r=blockIdx.x,k=blockIdx.y;float dot=0.f,square=0.f;
    int n=width*4;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float v=bf(x[r*n+i]);dot+=v*weight[k*n+i];square+=v*v;}
    dot=block_sum(dot);square=block_sum(square);
    if(threadIdx.x==0)mixes[r*24+k]=dot*rsqrtf(square/n+eps);
}

extern "C" __global__ void ca_hc_coefficients(const float* mixes,const float* scale,const float* base,
    float* pre,float* post,float* comb,int rows,float eps,int iters) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;
    const float* m=mixes+r*24;float c[16];
    for(int h=0;h<4;h++){
        pre[r*4+h]=logistic(m[h]*scale[0]+base[h])+eps;
        post[r*4+h]=2.f*logistic(m[4+h]*scale[1]+base[4+h]);
        float maxv=-INFINITY;
        for(int j=0;j<4;j++){c[h*4+j]=m[8+h*4+j]*scale[2]+base[8+h*4+j];maxv=fmaxf(maxv,c[h*4+j]);}
        float s=0.f;for(int j=0;j<4;j++){c[h*4+j]=expf(c[h*4+j]-maxv);s+=c[h*4+j];}
        for(int j=0;j<4;j++)c[h*4+j]=c[h*4+j]/s+eps;
    }
    for(int j=0;j<4;j++){float s=c[j]+c[4+j]+c[8+j]+c[12+j];for(int h=0;h<4;h++)c[h*4+j]/=s+eps;}
    for(int it=1;it<iters;it++){
        for(int h=0;h<4;h++){float s=c[h*4]+c[h*4+1]+c[h*4+2]+c[h*4+3];for(int j=0;j<4;j++)c[h*4+j]/=s+eps;}
        for(int j=0;j<4;j++){float s=c[j]+c[4+j]+c[8+j]+c[12+j];for(int h=0;h<4;h++)c[h*4+j]/=s+eps;}
    }
    for(int j=0;j<16;j++)comb[r*16+j]=c[j];
}

extern "C" __global__ void ca_swiglu(const float* gate,const float* up,__nv_bfloat16* out,int n,float limit) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float g=fminf(gate[i],limit),u=fminf(limit,fmaxf(-limit,up[i]));
    out[i]=__float2bfloat16_rn((g*logistic(g))*u);
}

extern "C" __global__ void ca_pool_pairs(const float* kv,const float* gate,__nv_bfloat16* out,int width) {
    int r=blockIdx.x;
    for(int i=threadIdx.x;i<width;i+=blockDim.x){int a=(r*2)*width+i,b=a+width;
        float m=fmaxf(gate[a],gate[b]),ga=expf(gate[a]-m),gb=expf(gate[b]-m);
        out[r*width+i]=__float2bfloat16_rn(kv[a]*(ga/(ga+gb))+kv[b]*(gb/(ga+gb)));
    }
}

extern "C" __global__ void ca_engram_gate(const __nv_bfloat16* x,const __nv_bfloat16* kv,
    const __nv_bfloat16* qw,const __nv_bfloat16* kw,__nv_bfloat16* out,int width,float eps) {
    int r=blockIdx.x,h=blockIdx.y;float xx=0.f,kk=0.f,dot=0.f;
    for(int i=threadIdx.x;i<width;i+=blockDim.x){int p=(r*4+h)*width+i;
        float a=bf(x[p]),b=bf(kv[(r*5+h)*width+i]);
        xx+=a*a;kk+=b*b;dot+=(a*(bf(qw[h*width+i])*bf(kw[h*width+i])))*b;
    }
    xx=block_sum(xx);kk=block_sum(kk);dot=block_sum(dot);
    dot=(dot*(rsqrtf(xx/width+eps)*rsqrtf(kk/width+eps)))*rsqrtf((float)width);
    float g=logistic(copysignf(sqrtf(fmaxf(fabsf(dot),1e-6f)),dot));
    for(int i=threadIdx.x;i<width;i+=blockDim.x){int p=(r*4+h)*width+i;
        out[p]=__float2bfloat16_rn(bf(x[p])+g*bf(kv[(r*5+4)*width+i]));}
}

// No approximate lookup: rows outside this rank's exact interval contribute zero.
extern "C" __global__ void ca_engram_dequant(const uint8_t* rows,const int64_t* ids,
    __nv_bfloat16* out,int count,int64_t row_start,int64_t row_end) {
    int r=blockIdx.x;if(r>=count)return;int64_t id=ids[r];
    for(int i=threadIdx.x;i<256;i+=blockDim.x){float v=0.f;
        if(id>=row_start && id<row_end){uint8_t q=rows[r*264+i],s=rows[r*264+256+i/32];
            float a=__half2float(__nv_cvt_fp8_to_halfraw(q,__NV_E4M3));
            float scale=s==255?NAN:(s==0?__int_as_float(1<<22):__int_as_float((int)s<<23));v=a*scale;}
        out[r*256+i]=__float2bfloat16_rn(v);
    }
}

// Reference frequency construction uses FP32 pow/div, a per-position FP32
// product, then CUDA sin/cos. Keep these boundaries; no host numeric table.
extern "C" __global__ void ca_rope_frequency_base(float* out,float base,int original_seq_len,float factor,float beta_fast,float beta_slow){
    int j=threadIdx.x;if(j>=32)return;
    float frequency=1.f/powf(base,(float)(j*2)/64.f);
    if(original_seq_len>0){
        constexpr double pi=3.1415926535897932384626433832795;
        double lb=log((double)base);
        float low=(float)fmax(0.,floor(64.*log((double)original_seq_len/((double)beta_fast*2.*pi))/(2.*lb)));
        float high=(float)fmin(63.,ceil(64.*log((double)original_seq_len/((double)beta_slow*2.*pi))/(2.*lb)));
        if(low==high)high+=.001f;
        float ramp=fminf(1.f,fmaxf(0.f,((float)j-low)*(1.f/(high-low))));
        float smooth=1.f-ramp;
        frequency=(frequency/factor)*(1.f-smooth)+frequency*smooth;
    }
    out[j]=frequency;
}
extern "C" __global__ void ca_rope_frequency_table(const float* frequencies,float* out,int rows){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*32)return;
    float angle=(float)(i/32)*frequencies[i%32];
    out[i*2]=cosf(angle);out[i*2+1]=sinf(angle);
}
// Non-completing/padded compressor rows never publish a latent to index-K.
// This is metadata masking only; valid BF16 latent values are untouched.
extern "C" __global__ void ca_mask_compressed_latent(__nv_bfloat16* latent,const int64_t* locations,int rows){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*512 && locations[i/512]==0)latent[i]=__ushort_as_bfloat16(0);
}

// Accepted encoder output is kept in a bounded128-token ring. Only byte copies
// occur here: the model's BF16/FP32 boundaries and source20 selection are intact.
extern "C" __global__ void ca_store_encoder_tail(
 const uint4* x,const float* pre,const int32_t* selected,const int32_t* lengths,
 const int64_t* tokens,const int32_t* positions,const int32_t* live,
 uint4* ring,float* ring_pre,int32_t* ring_ids,int32_t* ring_lens,int64_t* ring_tokens,
 int32_t* tags,uint32_t* error,int rows,int context){
 int v=blockIdx.x*blockDim.x+threadIdx.x;if(v>=rows*2560)return;
 int r=v/2560,j=v%2560;if(r>=live[0])return;int p=positions[r];
 if(p<0||p>=context){if(j==0)atomicOr(error,1u);return;}
 // Wide producers publish more than one ring lap: only the newest128 live
 // rows (consecutive positions) may own their slot. A no-op for <=128 rows.
 if(p<=positions[live[0]-1]-128)return;int slot=p%128;
 ring[slot*2560+j]=x[v];
 if(j<4)ring_pre[slot*4+j]=pre[r*4+j];
 if(j<512)ring_ids[slot*512+j]=selected[r*512+j];
 if(j==0){ring_lens[slot]=lengths[r];ring_tokens[slot]=tokens[r];tags[slot]=p;}
}
extern "C" __global__ void ca_gather_encoder_tail(
 const uint4* ring,const float* ring_pre,const int32_t* ring_ids,const int32_t* ring_lens,
 const int64_t* ring_tokens,const int32_t* tags,const int32_t* begin,
 uint4* out,float* out_pre,int32_t* out_ids,int32_t* out_lens,int64_t* out_tokens,
 uint32_t* error,int context){
 int v=blockIdx.x*blockDim.x+threadIdx.x;if(v>=128*2560)return;
 int r=v/2560,j=v%2560,p=begin[0]+r,slot=p%128;
 bool valid=p>=0&&p<context&&tags[slot]==p;
 out[v]=valid?ring[slot*2560+j]:make_uint4(0,0,0,0);
 if(j<4)out_pre[r*4+j]=valid?ring_pre[slot*4+j]:0.f;
 if(j<512)out_ids[r*512+j]=valid?ring_ids[slot*512+j]:-1;
 if(j==0){out_lens[r]=valid?ring_lens[slot]:0;out_tokens[r]=valid?ring_tokens[slot]:2;
  if(!valid)atomicOr(error,2u);}
}
// Two-level index selection, level one (canonical inference/model.py
// select_candidate_blocks): a block of 8 compressed positions scores as the max
// of its reachable positions; the block holding the newest position is pinned;
// unreachable blocks are -inf. Reads scores before top-k consumes them.
// `row0` is the first logical row held by a chunked score arena.
#include "candidate_block_scores.cuh"
#include "compact_candidate_scores.cuh"
#include "retained_candidate_publication.cuh"
#include "retained_candidate_gather.cuh"
#include "compact_retained_mask.cuh"
// Reachable blocks per row from the index windows, and how many are kept.
extern "C" __global__ void ca_candidate_block_windows(const int32_t* lengths,
 int32_t* block_lengths,int32_t* top_lengths,int rows,int topk){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;
 int len=lengths[r];int b=len>0?(len+7)/8:0;block_lengths[r]=b;top_lengths[r]=b<topk?b:topk;
}
// Kept blocks become a per-row byte mask (cleared by the caller each step).
extern "C" __global__ void ca_candidate_mask(const int32_t* top,int topk,
 const int32_t* top_lengths,uint8_t* mask,int mask_stride,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*topk)return;
 int r=i/topk,k=i%topk;if(k>=top_lengths[r])return;int b=top[i];
 if(b>=0&&b<mask_stride)mask[(size_t)r*mask_stride+b]=1;
}
// Level two for a consumer indexer: positions outside the published blocks
// cannot be selected. `mask_rows` offsets a chunked score arena's first row.
extern "C" __global__ void ca_apply_candidate_mask(float* scores,int stride,
 const int32_t* starts,const int32_t* lengths,const uint8_t* mask,int mask_stride,
 int mask_rows,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*stride)return;
 int r=i/stride,p=i%stride,m=mask_rows+r;if(p>=lengths[m])return;
 if(!mask[(size_t)m*mask_stride+p/8])scores[(size_t)r*stride+starts[m]+p]=-INFINITY;
}
#include "bounded_candidate_mask.cuh"

// Source20 candidate-block masks ride the same 128-slot ring as the encoder
// tail rows (tags are written by ca_store_encoder_tail in the same publish).
#include "bounded_encoder_tail.cuh"
extern "C" __global__ void ca_store_tail_mask(const uint8_t* mask,int stride,
 const int32_t* positions,const int32_t* live,uint8_t* ring,int rows,int context){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*stride)return;
 int r=i/stride,j=i%stride;if(r>=live[0])return;int p=positions[r];
 if(p<0||p>=context||p<=positions[live[0]-1]-128)return;
 ring[(size_t)(p%128)*stride+j]=mask[i];
}
extern "C" __global__ void ca_gather_tail_mask(const uint8_t* ring,int stride,
 const int32_t* tags,const int32_t* begin,uint8_t* out,int context){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=128*stride)return;
 int r=i/stride,j=i%stride,p=begin[0]+r,slot=p%128;
 bool valid=p>=0&&p<context&&tags[slot]==p;out[i]=valid?ring[(size_t)slot*stride+j]:0;
}
extern "C" __global__ void ca_reset_encoder_tail_tags(int32_t* tags){
 int i=threadIdx.x;if(i<128)tags[i]=-1;
}

// Native DSpark Markov latent lookup. The learned embedding is replicated;
// an invalid preceding token cannot become an out-of-bounds device read.
extern "C" __global__ void ca_dspark_markov_embed(const int32_t* token,
 const __nv_bfloat16* weight,__nv_bfloat16* latent,uint32_t* error){
 int i=threadIdx.x;int id=token[0];bool valid=id>=0&&id<129280;
 if(i==0)*error=valid?0u:1u;
 if(i<256)latent[i]=valid?weight[(size_t)id*256+i]:__ushort_as_bfloat16(0);
}
// Both canonical-head projections retain FP32 outputs, as in the published
// DeepSeek reference. The corrected distribution never rounds back to BF16.
extern "C" __global__ void ca_dspark_markov_local(const float* base,
 const float* bias,float* output,int step){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<32320)output[i]=base[(size_t)step*32320+i]+bias[i];
}

// One compact transfer publishes all five proposal IDs and both classes of
// validation flags. Raw integer copies preserve every error bit; no sampling
// decision or model arithmetic occurs here.
extern "C" __global__ void ca_dspark_markov_readout(
 const uint32_t* t0,const uint32_t* t1,const uint32_t* t2,const uint32_t* t3,const uint32_t* t4,
 const uint32_t* s0,const uint32_t* s1,const uint32_t* s2,const uint32_t* s3,const uint32_t* s4,
 const uint32_t* e0,const uint32_t* e1,const uint32_t* e2,const uint32_t* e3,const uint32_t* e4,
 uint32_t* out){
 const uint32_t* input[15]={t0,t1,t2,t3,t4,s0,s1,s2,s3,s4,e0,e1,e2,e3,e4};
 unsigned i=threadIdx.x;if(i<15)out[i]=input[i][0];if(i==15)out[i]=0x4453504bu;
}

// Bit-copy transpose of NCCL [rank,row,column] into model [row,rank,column].
// A pack is8 BF16/u16 values. No floating-point operations or conversion.
extern "C" __global__ void ca_transpose_rank_columns(
 const uint4* source,uint4* output,int rows,int column_packs) {
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 size_t count=(size_t)rows*4*column_packs;
 if(i<count){int col=i%column_packs;size_t q=i/column_packs;
  int rank=q%4;size_t row=q/4;
  output[i]=source[((size_t)rank*rows+row)*column_packs+col];}
}

// Canonical confidence features are post-HC (before final RMS) followed by
// the previous sampled token's Markov embedding. Widen original BF16 bytes
// exactly to FP32, matching the reference confidence head's materialization.
extern "C" __global__ void ca_dspark_confidence_weight(
 const __nv_bfloat16* source,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<5376)output[i]=__bfloat162float(source[i]);
}
extern "C" __global__ void ca_dspark_confidence_features(
 const __nv_bfloat16* post_hc,const __nv_bfloat16* previous,float* output,int step){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<5376)output[step*5376+i]=__bfloat162float(i<5120?post_hc[step*5120+i]:previous[i-5120]);
}
extern "C" __global__ void ca_dspark_confidence_probability(
 const float* projection,float* raw,float* probability,int step){
 if(threadIdx.x==0){float x=projection[0];raw[step]=x;
  probability[step]=1.f/(1.f+expf(-x));}
}
// Confidence controls work scheduling only. The target still verifies every
// proposed visible token. At least one draft is tested; at most five are used.
extern "C" __global__ void ca_dspark_confidence_limit(const float* probability,float* output){
 if(threadIdx.x==0){float cumulative=1.f;int count=0;bool valid=true;
  for(int i=0;i<5;++i){float p=probability[i];valid=valid&&isfinite(p)&&p>=0.f&&p<=1.f;
   cumulative*=p;if(cumulative>=0.1f)++count;}
  output[0]=valid?float(1+max(1,min(5,count))):0.f;
  output[1]=output[2]=output[3]=0.f;}
}
// Single-stream experiment: maximize predicted accepted inputs per measured
// complete step cost. Candidates3..6 have actual warm measurements; no
// fabricated1/2-row costs. Every visible proposal remains target-verified.
extern "C" __global__ void ca_dspark_confidence_cost_limit(
 const float* probability,const float* cost_ms,float* output){
 if(threadIdx.x==0){
  bool valid=true;float cumulative=1.f,expected=1.f,best=-1.f;int count=0;
  for(int i=0;i<5;++i){float p=probability[i];valid=valid&&isfinite(p)&&p>=0.f&&p<=1.f;
   cumulative*=p;expected+=cumulative;
   if(i>=1){float cost=cost_ms[i-1];valid=valid&&isfinite(cost)&&cost>0.f;
    float utility=expected/cost;if(utility>best){best=utility;count=i+2;}}}
  output[0]=valid?float(count):0.f;output[1]=output[2]=output[3]=0.f;
 }
}
// The first gathered pack belongs to TP rank0. Every physical peer uses its
// scheduling decision, avoiding divergent collective/graph live-row counts.
extern "C" __global__ void ca_dspark_confidence_readout(
 const float* raw,const float* probability,const float* gathered,uint32_t* output){
 int i=threadIdx.x;if(i<5){output[16+i]=__float_as_uint(raw[i]);output[21+i]=__float_as_uint(probability[i]);}
 if(i==0){float live=gathered[0];output[26]=(isfinite(live)&&live>=2.f&&live<=6.f&&live==floorf(live))?(uint32_t)live:0u;}
 if(i>=27&&i<31)output[i]=0u;if(i==31)output[31]=0x434f4e46u;
}

// s594: isolated parallel FP64 categorical finish.
#include "categorical_parallel.cuh"
#include "categorical_sharded.cuh"
#include "local_target_probabilities.cuh"
#include "target_transport_fused.cuh"
#include "target_selected_finish.cuh"
#include "hierarchical_categorical.cuh"
#include "hierarchy_local_fused.cuh"
#include "sharded_block.cuh"
#include "sharded_local_fused.cuh"
#include "target_probability_residual.cuh"

// s598: isolated FP64 probability normalization.
#include "categorical_normalize.cuh"

// Isolated next-round root-temperature feedback; target sampling is unchanged.
#include "root_feedback.cuh"

// Integer-only sampler result transport; no model or probability arithmetic.
#include "sampler_readout.cuh"

// Isolated fused block progress for next-round length feedback; target law unchanged.
#include "block_progress.cuh"

// Independent acceptance rows; each FP64 reduction keeps its serial order.
#include "block_decide_rows.cuh"

// Finite repeat diagnosis only: compare transported bits, never model values.
// Ordered launches select and preserve the earliest unequal frame without a
// host read or synchronization in the generating request.
extern "C" __global__ void ca_trace_first_difference(const uint32_t* current,
 const uint32_t* reference,uint32_t* first,int words,uint32_t index,int forced){
 if(*first<index)return;
 if(forced){if(blockIdx.x==0&&threadIdx.x==0)atomicMin(first,index);return;}
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=gridDim.x*blockDim.x)
  if(current[i]!=reference[i]){atomicMin(first,index);return;}
}
extern "C" __global__ void ca_trace_copy_first(const uint32_t* current,
 uint32_t* saved,const uint32_t* first,int words,uint32_t index){
 if(*first!=index)return;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=gridDim.x*blockDim.x)saved[i]=current[i];
}
#include "sampler_law_probe.cuh"
#include "key_sharded_index.cuh"
#include "key_sharded_publisher.cuh"
#include "allocation_placement_probe.cuh"

#include "block_finish_parallel.cuh"

#include "block_decide_finish_fused.cuh"

#include "ffn_sum_shared.cuh"
