// Diagnostic top-four IDs only. Probability calculations stay on the GPU.
extern "C" __global__ void ca_sampler_law_top4(const float* logits,int* output,int eos) {
    __shared__ float values[256];
    __shared__ int ids[256], chosen[4];
    int t=threadIdx.x;
    for(int k=0;k<4;++k) {
        float best=-INFINITY;int id=0x7fffffff;
        for(int i=t;i<129280;i+=256) {
            bool allowed=i!=eos;
            for(int j=0;j<k;++j)allowed=allowed&&i!=chosen[j];
            float v=logits[i];
            if(allowed&&(v>best||(v==best&&i<id))){best=v;id=i;}
        }
        values[t]=best;ids[t]=id;__syncthreads();
        for(int stride=128;stride;stride>>=1) {
            if(t<stride&&(values[t+stride]>values[t]||
                (values[t+stride]==values[t]&&ids[t+stride]<ids[t]))) {
                values[t]=values[t+stride];ids[t]=ids[t+stride];
            }
            __syncthreads();
        }
        if(!t){chosen[k]=ids[0];output[k]=ids[0];}
        __syncthreads();
    }
}
