#include "commoncudautils.h"
#include "commoncudautilshost.h"

NAMESPACE_BEGIN(krr)

template <bool tRcpSum>
void calcSum2PassAsync(const float *input, float *sum, float *partSum, size_t size,
					   CUstream stream) {
	const int2 bgSize = GetBlockSizeAndMinGridSize(SumTwoPassInterleavedKernelOp1<false>);
	const int32_t threadNumPerBlock = bgSize.x;
	const int32_t blockNum			= bgSize.y;

	// the first pass reduce input[0: n] to part[0: blockNum]
	// partSum[i] stands for the result of i-th block
	size_t shmSize = threadNumPerBlock * sizeof(float); // float per thread
	SumTwoPassInterleavedKernelOp1<false> CUDA_KERNEL(blockNum, threadNumPerBlock, shmSize,
													  stream)(input, partSum, size);

	// the second pass reduce part[0: blockNum] to output
	SumTwoPassInterleavedKernelOp1<tRcpSum> CUDA_KERNEL(1, threadNumPerBlock, shmSize,
														stream)(partSum, sum, blockNum);
}

// make template specialization
template void calcSum2PassAsync<true>(const float *input, float *sum, float *partSum, size_t size,
									  CUstream stream);
template void calcSum2PassAsync<false>(const float *input, float *sum, float *partSum, size_t size,
									   CUstream stream);

NAMESPACE_END(krr)