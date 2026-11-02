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
