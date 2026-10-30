// Selected DeepGEMM device kernel (MIT), PR324 14073b4e1e706506e193231209738c848d092a1f.
// Rust owns allocation, metadata, descriptors, lifetime and scheduling.
#include <cuda.h>
#include <cuda_runtime.h>
#include "mqa_keysplit.cuh"
using Map=CUtensorMap;
static_assert(sizeof(Map)==128);
static constexpr int SHARED=62064;
static auto kernel1=deep_gemm::sm120_fp4_mqa_logits_keysplit<32,128,true,4,128,2,5,48,128,256,float,1>;
static auto kernel4=deep_gemm::sm120_fp4_mqa_logits_keysplit<32,128,true,4,128,2,5,48,128,256,float,4>;
static auto kernel6=deep_gemm::sm120_fp4_mqa_logits_keysplit<32,128,true,4,128,2,5,48,128,256,float,6>;
static auto kernel8=deep_gemm::sm120_fp4_mqa_logits_keysplit<32,128,true,4,128,2,5,48,128,256,float,8>;
static auto kernel12=deep_gemm::sm120_fp4_mqa_logits_keysplit<32,128,true,4,128,2,5,48,128,256,float,12>;
static auto kernel16=deep_gemm::sm120_fp4_mqa_logits_keysplit<32,128,true,4,128,2,5,48,128,256,float,16>;
static auto kernel24=deep_gemm::sm120_fp4_mqa_logits_keysplit<32,128,true,4,128,2,5,48,128,256,float,24>;
extern "C" int ca_mqa_abi(){return 1;}
static int encode(Map* out,CUtensorMapDataType type,const void* ptr,uint64_t d0,uint64_t d1,
 uint64_t stride,uint32_t b0,uint32_t b1,CUtensorMapSwizzle swizzle){
 uint64_t dims[]={d0,d1};uint64_t strides[]={stride};uint32_t box[]={b0,b1},steps[]={1,1};
 return (int)cuTensorMapEncodeTiled(out,type,2,const_cast<void*>(ptr),dims,strides,box,steps,
   CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
extern "C" int ca_mqa_encode(int rows,int keys,const void* q,const void* qs,const void* k,const void* ks,
 const void* weights,void* maps){
 if(rows<1||rows>4096||keys<1||keys>1048576||((uintptr_t)maps)%64)return -1;
 auto m=(Map*)maps;int s;
 if((s=encode(m+0,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,q,128,rows*32,64,128,128,CU_TENSOR_MAP_SWIZZLE_64B)))return s;
 if((s=encode(m+1,CU_TENSOR_MAP_DATA_TYPE_INT32,qs,32,rows,128,32,4,CU_TENSOR_MAP_SWIZZLE_NONE)))return s;
 if((s=encode(m+2,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,k,128,keys,64,128,128,CU_TENSOR_MAP_SWIZZLE_64B)))return s;
 if((s=encode(m+3,CU_TENSOR_MAP_DATA_TYPE_INT32,ks,((keys+3)/4)*4,1,0,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)))return s;
 if((s=encode(m+4,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,weights,32,rows,128,32,4,CU_TENSOR_MAP_SWIZZLE_NONE)))return s;
 if((s=(int)cudaFuncSetAttribute((const void*)kernel1,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED)))return s;
if((s=(int)cudaFuncSetAttribute((const void*)kernel4,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED)))return s;
if((s=(int)cudaFuncSetAttribute((const void*)kernel6,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED)))return s;
if((s=(int)cudaFuncSetAttribute((const void*)kernel8,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED)))return s;
if((s=(int)cudaFuncSetAttribute((const void*)kernel12,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED)))return s;
if((s=(int)cudaFuncSetAttribute((const void*)kernel16,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED)))return s;
if((s=(int)cudaFuncSetAttribute((const void*)kernel24,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED)))return s;
 return 0;
}
extern "C" int ca_mqa_launch_split(uint32_t splits,uint32_t rows,uint32_t keys,uint32_t columns,uint32_t stride,
 const int32_t* starts,const int32_t* ends,float* logits,const void* maps,cudaStream_t stream){
 if(rows<1||rows>4096||keys<1||keys>1048576||columns<1||columns>keys||stride<columns||stride%128)return -1;
 auto m=(const Map*)maps;
 void* args[]={&rows,&keys,&columns,&stride,&starts,&ends,&logits,
  (void*)(m+0),(void*)(m+1),(void*)(m+2),(void*)(m+3),(void*)(m+4)};
 switch(splits){case 1:return (int)cudaLaunchKernel((const void*)kernel1,dim3(48),dim3(384),args,SHARED,stream);case 4:return (int)cudaLaunchKernel((const void*)kernel4,dim3(48),dim3(384),args,SHARED,stream);case 6:return (int)cudaLaunchKernel((const void*)kernel6,dim3(48),dim3(384),args,SHARED,stream);case 8:return (int)cudaLaunchKernel((const void*)kernel8,dim3(48),dim3(384),args,SHARED,stream);case 12:return (int)cudaLaunchKernel((const void*)kernel12,dim3(48),dim3(384),args,SHARED,stream);case 16:return (int)cudaLaunchKernel((const void*)kernel16,dim3(48),dim3(384),args,SHARED,stream);case 24:return (int)cudaLaunchKernel((const void*)kernel24,dim3(48),dim3(384),args,SHARED,stream);default:return -2;}
}
