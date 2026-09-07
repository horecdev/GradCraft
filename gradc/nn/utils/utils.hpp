#include <cuda_runtime.h>

inline cudaStream_t create_stream() {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    return stream;
}

inline cudaEvent_t create_event() {
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    return event;
}