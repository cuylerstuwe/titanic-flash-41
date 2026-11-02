// Finite prototype: preserve the existing127 global1024-token tiles while
// transporting tile statistics and the three tiles crossing TP4 boundaries.
// All exponentials, reduction trees, CDF order and uniforms remain unchanged.
// Full tiles have one owner; crossing tiles carry their original FP32 bytes.
constexpr int CA_SHARD_COLS=32320,CA_SHARD_WORDS=2176;
struct CaShardStat { double sum; float maximum; uint32_t error; };
static_assert(sizeof(CaShardStat)==16,"compact categorical ABI");

__device__ __forceinline__ bool ca_shard_boundary(int tile) {
  return tile==31||tile==63||tile==94;
}
__device__ __forceinline__ float ca_shard_boundary_value(
 const float* gathered,int row,int tile,int off) {
  int id=tile*1024+off,left=tile*1024/CA_SHARD_COLS,owner=id/CA_SHARD_COLS;
  int side=owner==left?1:0;
  return gathered[((size_t)row*4+owner)*CA_SHARD_WORDS+128+side*1024+off];
}
extern "C" __global__ void ca_sharded_categorical_pack(
 const float* local,float* packed,int rank,float temperature) {
  int row=blockIdx.y,slot=blockIdx.x,t=threadIdx.x;
  float* output=packed+(size_t)row*CA_SHARD_WORDS;
  int lower=rank*CA_SHARD_COLS,upper=lower+CA_SHARD_COLS;
  if(slot>=32) {
    int side=slot-32;
    int tile=side==0?lower/1024:upper/1024;
    bool present=side==0?rank>0:rank<3;
    for(int off=t;off<1024;off+=256) {
      int id=tile*1024+off;
      output[128+side*1024+off]=(present&&id>=lower&&id<upper)?
          local[(size_t)row*CA_SHARD_COLS+id-lower]:-INFINITY;
    }
    return;
  }
  int tile=(lower+1023)/1024+slot;
  int end=rank==3?127:upper/1024;
  auto* stats=reinterpret_cast<CaShardStat*>(output);
  if(tile>=end) {if(t==0)stats[slot]={0.,-INFINITY,0};return;}
  __shared__ float maximum;
  __shared__ uint32_t error;
  if(t==0)error=0;
  __syncthreads();
  CaGreedyBest best={-INFINITY,2147483647};
  for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=129280)continue;
    float x=local[(size_t)row*CA_SHARD_COLS+id-lower];
    if(isnan(x)||x==INFINITY){atomicOr(&error,1u);continue;}
    best=ca_greedy_better(best,CaGreedyBest{x,id});
  }
  best=ca_greedy_block(best);
  if(t==0)maximum=best.value;
  __syncthreads();
  double sum=0.;
  if(isfinite(maximum))for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=129280)continue;
    float x=local[(size_t)row*CA_SHARD_COLS+id-lower];
    if(isfinite(x))sum+=(double)expf((x-maximum)/temperature);
  }
  sum=ca_categorical_sum(sum);
  if(t==0)stats[slot]={sum,maximum,error};
}
extern "C" __global__ void ca_sharded_categorical_unpack(
 const float* gathered,float* maxima,double* sums,uint32_t* errors,float temperature) {
  int row=blockIdx.y,tile=blockIdx.x,t=threadIdx.x;
  if(!ca_shard_boundary(tile)) {
    if(t==0) {
      int owner=tile*1024/CA_SHARD_COLS;
      int slot=tile-(owner*CA_SHARD_COLS+1023)/1024;
      const auto* stats=reinterpret_cast<const CaShardStat*>(
          gathered+((size_t)row*4+owner)*CA_SHARD_WORDS);
      CaShardStat stat=stats[slot];
      maxima[row*127+tile]=stat.maximum;sums[row*127+tile]=stat.sum;
      if(stat.error)atomicOr(errors+row,stat.error);
    }
    return;
  }
  CaGreedyBest best={-INFINITY,2147483647};
  for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;
    float x=ca_shard_boundary_value(gathered,row,tile,off);
    if(isnan(x)||x==INFINITY){atomicOr(errors+row,1u);continue;}
    best=ca_greedy_better(best,CaGreedyBest{x,id});
  }
  best=ca_greedy_block(best);
  __shared__ float maximum;
  if(t==0){maximum=best.value;maxima[row*127+tile]=best.value;}
  __syncthreads();
  double sum=0.;
  if(isfinite(maximum))for(int off=t;off<1024;off+=256) {
    float x=ca_shard_boundary_value(gathered,row,tile,off);
    if(isfinite(x))sum+=(double)expf((x-maximum)/temperature);
  }
  sum=ca_categorical_sum(sum);
  if(t==0)sums[row*127+tile]=sum;
}
// Only the owner materializes the chosen tile. Reuse the exact established
// selected-tile CUB CDF kernel, with its original global token addressing.
extern "C" __global__ void ca_sharded_categorical_selected(
 const float* local,const float* gathered,const int32_t* selected,
 const uint32_t* errors,int32_t* owned,float* scratch,int rank) {
  int row=blockIdx.x,tile=selected[row],t=threadIdx.x;
  if(errors[row]||tile<0||tile>=127||tile*1024/CA_SHARD_COLS!=rank) {
    if(t==0)owned[row]=-1;return;
  }
  if(t==0)owned[row]=tile;
  for(int off=t;off<1024;off+=256) {
    int id=tile*1024+off;if(id>=129280)continue;
    scratch[(size_t)row*129280+id]=ca_shard_boundary(tile)?
        ca_shard_boundary_value(gathered,row,tile,off):
        local[(size_t)row*CA_SHARD_COLS+id-rank*CA_SHARD_COLS];
  }
}
extern "C" __global__ void ca_sharded_categorical_id_pack(
 const int32_t* ids,const uint32_t* errors,float* packed,int rows) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
  uint32_t* out=reinterpret_cast<uint32_t*>(packed)+row*4;
  out[0]=(uint32_t)ids[row];out[1]=errors[row];out[2]=0;out[3]=0;
}
extern "C" __global__ void ca_sharded_categorical_id_finish(
 const float* gathered,int32_t* ids,uint32_t* errors,int rows) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
  const uint32_t* in=reinterpret_cast<const uint32_t*>(gathered)+row*16;
  int count=0,id=-1;uint32_t error=0;
  for(int rank=0;rank<4;++rank) {
    int candidate=(int32_t)in[rank*4];error|=in[rank*4+1];
    if(candidate>=0&&candidate<129280){++count;id=candidate;}
  }
  if(!error&&count!=1)error|=64u;
  errors[row]=error;ids[row]=error?-1:id;
}

// GPU-only diagnostic inputs. No host model arithmetic or CPU sampler oracle.
extern "C" __global__ void ca_sharded_categorical_fixture(float* local,int rank,int rows,int phase) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*CA_SHARD_COLS)return;
  int row=i/CA_SHARD_COLS,id=rank*CA_SHARD_COLS+i%CA_SHARD_COLS;
  const int edges[22]={0,31743,31744,32319,32320,32767,32768,64511,64512,64639,64640,
      65535,65536,96255,96256,96959,96960,97279,97280,129023,129024,129279};
  uint32_t h=uint32_t(id)*0x9e3779b9u+uint32_t(row+1)*0x6d2b79f5u;
  float x=float(int(h%2001)-1000)*0.0078125f;
  if(phase==1){x=-INFINITY;for(int k=0;k<22;++k)if(id==edges[k])x=float(k%5-2)*0.5f;}
  if(phase==2)x=id==edges[row%22]?0.f:-INFINITY;
  if(phase==3)x=(h&15)==0?-INFINITY:float(int(h%36001)-18000)*0.0078125f;
  if(phase==4)x=-INFINITY;
  if(phase==5&&id==edges[row%22])x=NAN;
  if(phase==6&&id==edges[row%22])x=INFINITY;
  if(phase==7){x=(id&1)?-0.f:0.f;if(row%4==1)x=-INFINITY;
      if(row%4==2&&id==32320)x=NAN;if(row%4==3&&id==96960)x=INFINITY;}
  local[i]=x;
}
