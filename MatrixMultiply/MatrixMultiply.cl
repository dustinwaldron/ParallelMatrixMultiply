// Modified from Intel BitonicSort example

__kernel void __attribute__((vec_type_hint(float))) MatrixMultiply(__global float *matrixA,
						__global float *matrixB,
						__global float *result,
						 const uint matrixSize)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	float sum = 0.0;

	__global float *nextA = matrixA + i*matrixSize;
	__global float *nextB = matrixB + j;

	// TODO: Loop to multiply row and column, should be about the same as the loop in the C++ code
	int k;
	for(k=0;k<matrixSize;k++){
		//float a = *(nextA + k);
		//float b = *(nextB + k*matrixSize);
		//sum = sum + (a*b);
		sum = sum + matrixA[i*matrixSize + k]*matrixB[k*matrixSize + j];
	}
	
	result[i*matrixSize + j] = sum;
}