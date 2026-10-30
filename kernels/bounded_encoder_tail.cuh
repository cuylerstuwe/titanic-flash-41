// Byte-only bounded publication, screened against the original kernels in
// s397. Producers supply consecutive positions; only the newest128 survive.
extern "C" __global__ void ca_store_encoder_tail_bounded(
 const uint4* x,const float* pre,const int32_t* selected,const int32_t* lengths,
 const int64_t* tokens,const int32_t* positions,const int32_t* live,
 uint4* ring,float* ring_pre,int32_t* ring_ids,int32_t* ring_lens,int64_t* ring_tokens,
 int32_t* tags,uint32_t* error,int rows,int context){
 int v=blockIdx.x*blockDim.x+threadIdx.x;
 // Do not hide invalid positions in discarded producer rows. For admitted
 // <=8192-row profiles this bounded grid still has at least one lane per row.
 if(v<rows&&v<live[0]){
  int p=positions[v];if(p<0||p>=context)atomicOr(error,1u);
 }
 if(v>=min(rows,128)*2560)return;
 int r=max(0,live[0]-128)+v/2560,j=v%2560;
 if(r>=live[0])return;
 int p=positions[r];
 if(p<0||p>=context||p<=positions[live[0]-1]-128)return;
 int slot=p%128;
 ring[slot*2560+j]=x[(size_t)r*2560+j];
 if(j<4)ring_pre[slot*4+j]=pre[r*4+j];
 if(j<512)ring_ids[slot*512+j]=selected[r*512+j];
 if(j==0){ring_lens[slot]=lengths[r];ring_tokens[slot]=tokens[r];tags[slot]=p;}
}

extern "C" __global__ void ca_store_tail_mask_bounded(
 const uint8_t* mask,int stride,const int32_t* positions,const int32_t* live,
 uint8_t* ring,int rows,int context){
 // The host requires16-byte aligned pointers and row strides.
 int width=stride/16;
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=min(rows,128)*width)return;
 int r=max(0,live[0]-128)+i/width,j=i%width;
 if(r>=live[0])return;
 int p=positions[r];
 if(p<0||p>=context||p<=positions[live[0]-1]-128)return;
 reinterpret_cast<uint4*>(ring)[(size_t)(p%128)*width+j]=
  reinterpret_cast<const uint4*>(mask)[(size_t)r*width+j];
}
