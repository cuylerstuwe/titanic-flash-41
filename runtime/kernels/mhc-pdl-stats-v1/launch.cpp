#include <cuda.h>
// The CUDA header owns the extensible launch ABI; Python passes only ordinary
// scalar dimensions and the existing AOT parameter-pointer vector.
extern "C" int launch_pair_op(CUfunction function, unsigned int x, unsigned int y,
    unsigned int threads, unsigned int shared, CUstream stream, void** parameters,
    int pdl) {
    CUlaunchConfig cfg = {};
    cfg.gridDimX=x; cfg.gridDimY=y; cfg.gridDimZ=1;
    cfg.blockDimX=threads; cfg.blockDimY=1; cfg.blockDimZ=1;
    cfg.sharedMemBytes=shared; cfg.hStream=stream;
    CUlaunchAttribute attribute = {};
    attribute.id=CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attribute.value.programmaticStreamSerializationAllowed=1;
    if(pdl) {cfg.attrs=&attribute;cfg.numAttrs=1;}
    return (int)cuLaunchKernelEx(&cfg,function,parameters,nullptr);
}
