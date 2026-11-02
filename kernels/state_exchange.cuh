// Lossless two-owner exchange. Each row owns disjoint, 16-byte-aligned live
// and inactive byte ranges. The host validates both layouts before launch.
// Prefix lengths may differ: never read an unpublished tail from either side.
extern "C" __global__ void ca_state_exchange_prefix(const uint64_t* plan) {
    const uint64_t* row = plan + uint64_t(blockIdx.y) * 4;
    uint4* live = reinterpret_cast<uint4*>(row[0]);
    uint4* inactive = reinterpret_cast<uint4*>(row[1]);
    const uint64_t save = row[2], restore = row[3];
    const uint64_t count = save > restore ? save : restore;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += uint64_t(gridDim.x) * blockDim.x) {
        uint4 a, b;
        if (i < save) a = live[i];
        if (i < restore) b = inactive[i];
        if (i < save) inactive[i] = a;
        if (i < restore) live[i] = b;
    }
}
