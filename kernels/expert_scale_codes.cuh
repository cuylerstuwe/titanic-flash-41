// Exact four-bit row deltas for the canonical E8M0 scale byte layout.
// Independent GPU packer; unsupported spans fail closed before model use.
// The initial full-model probe retains original strides and raw operands.
extern "C" __global__ void ca_expert_scale_codes4(
 const uint8_t* source,uint8_t* output,uint32_t* error,int matrices){
 const int matrix=blockIdx.y,row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
 if(matrix>=matrices||row>=576)return;
 const int tile=row/128,nr=min(128,576-tile*128),r=row%128;
 const int source_base=matrix*92160+tile*128*160;
 int lo=255,hi=0;
 for(int k=lane;k<160;k+=32){
  int x=source[source_base+(k/4)*nr*4+r*4+k%4];lo=min(lo,x);hi=max(hi,x);
 }
 lo=__reduce_min_sync(0xffffffff,lo);hi=__reduce_max_sync(0xffffffff,hi);
 if(hi>247||hi-lo>15){if(lane==0)atomicOr(error,1u);return;}
 const int destination=matrix*92160;
 if(lane==0){output[destination+16+row]=(uint8_t)lo;if(row==0)output[destination]=4;}
 for(int pair=lane;pair<80;pair+=32){
  int k=pair*2;
  int a=source[source_base+(k/4)*nr*4+r*4+k%4];
  int b=source[source_base+(k/4)*nr*4+r*4+k%4+1];
  uint8_t packed=(uint8_t)((a-lo)|((b-lo)<<4));
  if(lo+(packed&15)!=a||lo+(packed>>4)!=b)atomicOr(error,2u);
  int target=destination+16+576+tile*128*80+(pair/2)*nr*2+r*2+pair%2;
  output[target]=packed;
 }
}

// One-bit scale code (Nov 18 2026; idea: pico's dsk::scale_code, MIT; layout ours). Input: one compact expert scale
// matrix per block (ca_expert_compact_scales layout: [128-row tile][K128 slice][row][width bytes], width 4, or 2 in a
// K tail). Output per matrix: a base word b * 0x01010101, where b is the matrix's commonest value, or the value below
// it when that one is more common than the value above; per (tile, slice) unit, 16 code words (bit 4 (r % 8) + j of
// word r / 8 = byte j of row r minus b) and 4 flag words (bit r % 32 of word r / 32: row r holds a byte outside
// {b, b + 1}, so the kernels read its raw word). Rows past the tile's rows, bytes past the width and flagged rows code 0.
// The kernels stage one unit per (tile, slice) as the raw path stages its 128 row words.
__device__ __forceinline__ int sc1_tiles(int n){return (n+127)/128;}
__device__ __forceinline__ int sc1_slices(int columns){return (columns+3)/4;}
extern "C" __global__ void ca_expert_scale_code1(
 const uint8_t* scales,uint32_t* code,uint32_t* flags,uint32_t* bases,int n,int columns){
 __shared__ unsigned hist[256];
 __shared__ int chosen;
 const int matrix=blockIdx.x,tiles=sc1_tiles(n),slices=sc1_slices(columns);
 const uint8_t* s=scales+(size_t)matrix*n*columns;
 for(int i=threadIdx.x;i<256;i+=blockDim.x)hist[i]=0;
 __syncthreads();
 for(int i=threadIdx.x;i<n*columns;i+=blockDim.x)atomicAdd(&hist[s[i]],1u);
 __syncthreads();
 if(threadIdx.x==0){
  int c=0;for(int v=1;v<256;v++)if(hist[v]>hist[c])c=v;
  int b=(c>0&&(c==255||hist[c-1]>hist[c+1]))?c-1:c;
  chosen=b;bases[matrix]=(uint32_t)b*0x01010101u;
 }
 __syncthreads();
 const unsigned b=(unsigned)chosen;
 const size_t units=(size_t)tiles*slices;
 uint32_t* c=code+(size_t)matrix*units*16;
 uint32_t* f=flags+(size_t)matrix*units*4;
 // One thread per (unit, 32-row group): four code words and one flag word.
 for(size_t g=threadIdx.x;g<units*4;g+=blockDim.x){
  const int unit=(int)(g/4),group=(int)(g%4),tile=unit/slices,slice=unit%slices;
  const int nr=min(128,n-tile*128),width=min(4,columns-slice*4);
  const uint8_t* u=s+(size_t)tile*128*columns+(size_t)slice*nr*4;
  uint32_t words[4]={0,0,0,0},flag=0;
  for(int i=0;i<32;i++){
   const int r=group*32+i;
   if(r>=nr)continue;
   unsigned nibble=0;bool raw=false;
   for(int j=0;j<width;j++){
    const unsigned v=u[r*width+j];
    if(v==b)continue;
    if(v==b+1){nibble|=1u<<j;continue;}
    raw=true;
   }
   if(raw){flag|=1u<<i;continue;}
   words[i/8]|=nibble<<(4*(i%8));
  }
  for(int w=0;w<4;w++)c[(size_t)unit*16+group*4+w]=words[w];
  f[(size_t)unit*4+group]=flag;
 }
}
// The kernels' decode, checked against every row word (error bit 1: a coded row decodes wrong; bit 2: a flagged row
// is fully on the pair, which the encoder never produces): word = flagged ? raw : ((n * 0x00204081) & 0x01010101) + base,
// masked to the width. Rows past a tile's rows are not checked (the kernels stage zeros there, as the raw path does).
extern "C" __global__ void ca_expert_scale_code1_check(
 const uint8_t* scales,const uint32_t* code,const uint32_t* flags,const uint32_t* bases,uint32_t* error,int n,int columns){
 const int matrix=blockIdx.y,tiles=sc1_tiles(n),slices=sc1_slices(columns);
 const size_t units=(size_t)tiles*slices;
 const size_t row_id=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(row_id>=units*128)return;
 const int unit=(int)(row_id/128),r=(int)(row_id%128),tile=unit/slices,slice=unit%slices;
 const int nr=min(128,n-tile*128),width=min(4,columns-slice*4);
 if(r>=nr)return;
 const uint8_t* u=scales+(size_t)matrix*n*columns+(size_t)tile*128*columns+(size_t)slice*nr*4+(size_t)r*width;
 uint32_t raw=0;for(int j=0;j<width;j++)raw|=(uint32_t)u[j]<<(8*j);
 const uint32_t mask=width==4?0xffffffffu:(1u<<(8*width))-1u;
 const uint32_t flag=(flags[((size_t)matrix*units+unit)*4+r/32]>>(r%32))&1u;
 const uint32_t word=code[((size_t)matrix*units+unit)*16+r/8];
 const uint32_t nibble=(word>>(4*(r%8)))&15u;
 const uint32_t decoded=(((nibble*0x00204081u)&0x01010101u)+bases[matrix])&mask;
 if(!flag&&decoded!=raw)atomicOr(error,1u);
 if(flag){
  const uint32_t b=bases[matrix]&255u;bool on=true;
  for(int j=0;j<width;j++){const uint32_t v=(raw>>(8*j))&255u;on=on&&(v==b||v==b+1);}
  if(on)atomicOr(error,2u);
 }
}
// Census of a coded layer: flagged rows and coded rows (two counters), for the load report.
extern "C" __global__ void ca_expert_scale_code1_census(const uint32_t* flags,unsigned long long* counts,size_t words){
 const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=words)return;
 const unsigned bits=__popc(flags[i]);
 if(bits)atomicAdd(counts,(unsigned long long)bits);
}
