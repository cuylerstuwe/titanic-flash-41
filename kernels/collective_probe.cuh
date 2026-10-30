// GPU-only fixtures/oracle for the isolated transport screen. No model path
// calls these functions. Both transports receive identical canonical BF16 bits.
__device__ __forceinline__ uint32_t ca_collective_hash(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
  x *= 0x846ca68bu; return x ^ (x >> 16);
}
__device__ __forceinline__ float ca_collective_value(int i,int rank,int pattern) {
  uint32_t h=ca_collective_hash(uint32_t(i)+uint32_t(rank)*0x9e3779b9u+37u);
  float mantissa=float(int(h&511u)-256);
  if(pattern==0)return __bfloat162float(__float2bfloat16_rn(mantissa/256.f));
  if(pattern==1)return __bfloat162float(__float2bfloat16_rn(ldexpf(mantissa,int((h>>10)%41)-28)));
  // Deliberate large/small cancellation exercises admissible order differences.
  float large=ldexpf(float(128+(ca_collective_hash(i)&127u)),12);
  if(rank==0)return large;if(rank==2)return -large;
  return __bfloat162float(__float2bfloat16_rn(mantissa/1024.f));
}
extern "C" __global__ void ca_collective_fixture(__nv_bfloat16* x,int n,int rank,int pattern) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)x[i]=__float2bfloat16_rn(ca_collective_value(i,rank,pattern));
}
extern "C" __global__ void ca_probe_f32_to_bf16(const float* x,__nv_bfloat16* y,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)y[i]=__float2bfloat16_rn(x[i]);
}
extern "C" __global__ void ca_collective_check(const __nv_bfloat16* y,
 const float* unrounded,uint32_t* stats,int n,int pattern,int has_f32) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  double exact=0.,scale=0.;
  for(int r=0;r<4;++r){double v=double(ca_collective_value(i,r,pattern));exact+=v;scale+=fabs(v);}
  // Conservative bound for three FP32 additions, followed by BF16 RN.
  // Negative mode admits the baseline's BF16 intermediate reductions. All
  // native/FP32 candidate modes keep the original much tighter FP32 bound.
  double fp32_bound=8.*(has_f32<0?0x1p-8:0x1p-24)*scale;
  float actual=__bfloat162float(y[i]);
  double rounded_bound=fp32_bound+fabs(double(actual))*0x1p-8+0x1p-133;
  if(!isfinite(actual))atomicAdd(stats+0,1u);
  else if(fabs(double(actual)-exact)>rounded_bound)atomicAdd(stats+1,1u);
  float reference=__bfloat162float(__float2bfloat16_rn(float(exact)));
  if(actual!=reference)atomicAdd(stats+2,1u);
  if(actual!=0.f)atomicAdd(stats+3,1u);
  if(has_f32>0){float raw=unrounded[i];
    if(!isfinite(raw)||fabs(double(raw)-exact)>fp32_bound)atomicAdd(stats+4,1u);
  }
}

// Changing payloads expose stale slots/stripes that repeated constant fixtures
// cannot. All values and rank-ordered reference arithmetic are GPU generated.
__device__ __forceinline__ float ca_sequence_value(int i,int rank,uint32_t epoch,int pattern) {
 uint32_t h=ca_collective_hash(uint32_t(i)^((uint32_t(rank)+1)*0x9e3779b9u)^((epoch+1)*0x85ebca6bu));
 float x=float(int(h&255u)-128);
 if(pattern) x=ldexpf(x,int((h>>8)%25)-20);
 return __bfloat162float(__float2bfloat16_rn(x));
}
extern "C" __global__ void ca_sequence_fill(__nv_bfloat16* x,const uint32_t* epoch,
 int n,int rank,int pattern,int jitter) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i==0&&jitter&&((*epoch&3u)==uint32_t(rank))){
  uint64_t start=clock64();while(clock64()-start<120000u){}
 }
 if(i<n)x[i]=__float2bfloat16_rn(ca_sequence_value(i,rank,*epoch,pattern));
}
extern "C" __global__ void ca_sequence_check(const __nv_bfloat16* y,const uint32_t* epoch,
 uint32_t* stats,int n,int pattern,int gather) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;int count=gather?n*4:n;if(i>=count)return;
 float expected;
 if(gather){int row=i/(4*5120),r=(i/5120)%4,h=i%5120;
  expected=ca_sequence_value(row*5120+h,r,*epoch,pattern);
 }else{
  expected=0.f;
  for(int r=0;r<4;++r)expected=__fadd_rn(expected,ca_sequence_value(i,r,*epoch,pattern));
 }
 uint16_t a=__bfloat16_as_ushort(y[i]);
 uint16_t b=__bfloat16_as_ushort(__float2bfloat16_rn(expected));
 if(a!=b){atomicAdd(stats,1u);atomicMin(stats+1,*epoch);
  if(atomicCAS(stats+2,0xffffffffu,uint32_t(i))==0xffffffffu){stats[3]=a;stats[4]=b;stats[5]=*epoch;stats[6]=gather;}
 }
}
extern "C" __global__ void ca_sequence_advance(uint32_t* epoch){if(threadIdx.x==0)++*epoch;}
