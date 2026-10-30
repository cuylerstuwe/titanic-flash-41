// Source20 encoder-only byte publication. Compact slots represent the newest
// 128 live rows, not global producer rows. The s946 component screen validated
// this layout, changed-input graph replay and untouched ring rows. Clearing
// and setting remain separate ordered launches on the target's common stream.
extern "C" __global__ void ca_zero_compact_tail_mask(
 uint8_t* mask,int stride,int rows,const int32_t* live){
 int width=stride/16,i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=128*width)return;
 int slot=i/width,g=max(0,live[0]-128)+slot;
 if(g>=live[0]||g>=rows)return;
 reinterpret_cast<uint4*>(mask)[(size_t)slot*width+i%width]=make_uint4(0,0,0,0);
}

extern "C" __global__ void ca_build_compact_tail_mask_packed(
 const int32_t* packed,const int32_t* top,const int32_t* live,
 uint8_t* mask,int stride,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=128*2048)return;
 int slot=i/2048,k=i%2048,g=max(0,live[0]-128)+slot;
 if(g>=live[0]||g>=rows||k>=top[g])return;
 int owner=g/(rows/4),b=packed[((size_t)owner*128+slot)*2048+k];
 if(b>=0&&b<stride)mask[(size_t)slot*stride+b]=1;
}

// The512-row profile already gathers its full quarter without packing. This
// variant also preserves the explicit full-ID-gather diagnostic control.
extern "C" __global__ void ca_build_compact_tail_mask(
 const int32_t* ids,const int32_t* top,const int32_t* live,
 uint8_t* mask,int stride,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=128*2048)return;
 int slot=i/2048,k=i%2048,g=max(0,live[0]-128)+slot;
 if(g>=live[0]||g>=rows||k>=top[g])return;
 int b=ids[(size_t)g*2048+k];
 if(b>=0&&b<stride)mask[(size_t)slot*stride+b]=1;
}

extern "C" __global__ void ca_store_compact_tail_mask(
 const uint8_t* mask,int stride,const int32_t* positions,const int32_t* live,
 uint8_t* ring,int rows,int context){
 int width=stride/16,i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=128*width)return;
 int slot=i/width,g=max(0,live[0]-128)+slot;
 if(g>=live[0]||g>=rows)return;
 int p=positions[g];
 if(p<0||p>=context||p<=positions[live[0]-1]-128)return;
 reinterpret_cast<uint4*>(ring)[(size_t)(p%128)*width+i%width]=
  reinterpret_cast<const uint4*>(mask)[(size_t)slot*width+i%width];
}
