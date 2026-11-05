// Raw-byte snapshot copies. All source/destination regions are independent;
// the caller owns their lifetimes and orders the launch after their producers.
// Bounded launch arguments avoid a descriptor allocation/upload per snapshot.
struct StateCopyRegion { uint64_t source, destination, bytes; };
struct StateCopyPlan { StateCopyRegion regions[128]; uint32_t end_blocks[128]; };
static_assert(sizeof(StateCopyRegion)==24 && sizeof(StateCopyPlan)==3584);
extern "C" __global__ void ca_state_copy_plan(const __grid_constant__ StateCopyPlan plan,
                                             uint32_t count) {
 uint32_t lo=0,hi=count;
 while(lo<hi) {uint32_t mid=(lo+hi)/2;if(blockIdx.x<plan.end_blocks[mid])hi=mid;else lo=mid+1;}
 if(lo==count)return;
 const StateCopyRegion r=plan.regions[lo];
 const uint64_t first=uint64_t(blockIdx.x-(lo?plan.end_blocks[lo-1]:0))*131072;
 const uint64_t bytes=(r.bytes-first)<131072?(r.bytes-first):131072;
 const uint8_t* source=reinterpret_cast<const uint8_t*>(r.source+first);
 uint8_t* destination=reinterpret_cast<uint8_t*>(r.destination+first);
 if(((r.source|r.destination|bytes)&15)==0) {
  const uint4* s=reinterpret_cast<const uint4*>(source);
  uint4* d=reinterpret_cast<uint4*>(destination);
  for(uint64_t i=threadIdx.x;i<bytes/16;i+=blockDim.x)d[i]=s[i];
 }else{
  for(uint64_t i=threadIdx.x;i<bytes;i+=blockDim.x)destination[i]=source[i];
 }
}
