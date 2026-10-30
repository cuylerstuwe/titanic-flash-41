// Test harness allocator: every synthetic fixture allocation starts initialized.
// It is not part of the inference runtime or its performance measurements.
#include <cuda_runtime.h>
#include <sys/types.h>
#include <cstdint>
#include <map>
#include <mutex>
static std::mutex mutex;
static std::map<uintptr_t,size_t> allocations;
extern "C" void* ca_fixture_alloc(ssize_t size,int device,cudaStream_t stream){
    if(cudaSetDevice(device)!=cudaSuccess)return nullptr;
    void* p=nullptr;if(cudaMalloc(&p,(size_t)size)!=cudaSuccess)return nullptr;
    if(cudaMemsetAsync(p,0,(size_t)size,stream)!=cudaSuccess){cudaFree(p);return nullptr;}
    {std::lock_guard<std::mutex> guard(mutex);allocations[(uintptr_t)p]=(size_t)size;}
    return p;
}
extern "C" void ca_fixture_free(void* p,ssize_t,int device,cudaStream_t){
    cudaSetDevice(device);cudaFree(p);
    std::lock_guard<std::mutex> guard(mutex);allocations.erase((uintptr_t)p);
}
extern "C" int ca_fixture_lookup(uint64_t ptr,uint64_t* base,uint64_t* size){
    std::lock_guard<std::mutex> guard(mutex);
    auto it=allocations.upper_bound((uintptr_t)ptr);
    if(it==allocations.begin())return 0;
    --it;if(ptr-it->first>=it->second)return 0;
    *base=it->first;*size=it->second;return 1;
}
