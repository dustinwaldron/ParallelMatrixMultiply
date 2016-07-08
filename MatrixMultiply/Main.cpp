#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>
#include <process.h>
#include <math.h>
#include "utils.h"

bool g_RunOnGPU = true;

char		*g_wantedName = NULL;
cl_mem		g_inputBuffer = NULL;
cl_context	g_context = NULL;
cl_command_queue g_cmd_queue = NULL;
cl_program	g_program = NULL;
cl_kernel	g_kernel = NULL;

#include <Windows.h>

LARGE_INTEGER g_PerfFrequency;
LARGE_INTEGER g_PerformanceCountNDRangeStart;
LARGE_INTEGER g_PerformanceCountNDRangeStop;

#define ARRAY_SIZE 1000

typedef struct _threadData
{
	cl_float* matrixA;
	cl_float* matrixB;
	cl_float* result;
	// TODO: Add other values needed for partiioning data among threads
	//matrixMultiplyWithThreads(nbrThreads, matrixA, matrixB, matrixResultThreads);
	int partition_size;
	int start;

} ThreadData;

void generateInput(cl_float* matrix)
{
    // random initialization of input
    for (size_t i = 0; i < ARRAY_SIZE * ARRAY_SIZE; ++i)
    {
        matrix[i] = (cl_float)((float) (rand() / (float) RAND_MAX));
    }
}

void multiplyRowCol(cl_float* matrixA, int row, cl_float* matrixB, int col, cl_float* result)
{
	// TODO: Add code to multiply a row of matrixA and a column of matrixB
	// NOTE: Data is stored by row - it will be most efficient to use pointers to access data
	// you can just increment the row pointer, but need to add ARRAY_SIZE to the column pointer
	// C[i][j] += A[i][k] * B[k][j]
	int k = 0;
	cl_float sum = 0;
	while(k < ARRAY_SIZE){
		sum = sum + matrixA[(ARRAY_SIZE*row) + k] * matrixB[(ARRAY_SIZE*k) + col];
		k=k+1;
	}
	result[(ARRAY_SIZE*row) + col ] = sum;
}

void matrixMultiply(cl_float* matrixA, cl_float* matrixB, cl_float* result)
{
	for (int i = 0; i < ARRAY_SIZE; i++)
		for (int j = 0; j < ARRAY_SIZE; j++)
			multiplyRowCol(matrixA, i, matrixB, j, result);
}

void threadMultiplyRowCol(void* args)
{
	ThreadData* myData = (ThreadData*) args;
	cl_float* A = myData->matrixA;
	cl_float* B = myData->matrixB;
	cl_float* C = myData->result;
	int size = myData->partition_size;
	int start = myData->start;

	// TODO: need to add code to multiply all rows and columns for the partition handled by this thread
	cl_float sum = 0;
	for(int row= start;row<start+size;row++){
		for(int col = start ;col<start+size;col++){
			multiplyRowCol(A,row,B,col,C);
		}
	}

	// notify that thread is done - good practice, but not required
	_endthread();
}

void matrixMultiplyWithThreads(int nbrThreads, cl_float* matrixA, cl_float* matrixB, cl_float* result)
{
	ThreadData* threadArgs;
	threadArgs = (ThreadData*)malloc(sizeof(ThreadData)*nbrThreads);
	HANDLE* threadHandles;
	threadHandles = (HANDLE*)malloc(sizeof(HANDLE)*nbrThreads);
	for (int i = 0; i < nbrThreads; i++)
	{
		for(int threadId=0; threadId < i+1; threadId++)
		{ 
			bool notNicePartitions = false;
			threadArgs[i].matrixA = matrixA;
			threadArgs[i].matrixB = matrixB;
			threadArgs[i].result = result;
			// TODO: initialize other values needed for partitioning data 
			int displacement = ARRAY_SIZE % (i+1);
			if(displacement != 0 && i != 0) 
			{
				notNicePartitions = true;
			}
			threadArgs[i].partition_size = ARRAY_SIZE/(i+1);
			if(threadId == i && notNicePartitions) 
			{
				threadArgs[i].partition_size = threadArgs[i].partition_size + displacement;
			}
			threadArgs[i].start = threadId*(ARRAY_SIZE/(i+1));
			threadHandles[i] = (HANDLE)_beginthread(threadMultiplyRowCol, 0, threadArgs + i);
		}
	}
	// wait for all threads
	DWORD waitResult = WaitForMultipleObjects(nbrThreads, threadHandles, true, INFINITE);
	if (waitResult != WAIT_OBJECT_0)
		printf("Wait failed, result = %x\n", waitResult);
	free(threadArgs);
	free(threadHandles);
}

void Cleanup_OpenCL()
{
    if( g_inputBuffer ) {clReleaseMemObject( g_inputBuffer ); g_inputBuffer = NULL;}
    if( g_kernel ) {clReleaseKernel( g_kernel ); g_kernel = NULL;}
    if( g_program ) {clReleaseProgram( g_program ); g_program = NULL;}
    if( g_cmd_queue ) {clReleaseCommandQueue( g_cmd_queue ); g_cmd_queue = NULL;}
    if( g_context ) {clReleaseContext( g_context ); g_context = NULL;}
}

int Setup_OpenCL( const char *program_source, cl_uint* alignment)
{
    cl_device_id devices[16];
    size_t cb;
    cl_uint size_ret = 0;
    cl_int err;

	if(g_RunOnGPU)
	{
		printf("Trying to run on a GPU \n");
	}
	else
	{
		printf("Trying to run on a CPU \n");
	}

	cl_platform_id intel_platform_id = GetOCLPlatform(g_wantedName);
    if( intel_platform_id == NULL )
    {
        printf("ERROR: Failed to find Intel OpenCL platform.\n");
        return -1;
    }

	// list devices available on platform
	cl_uint nbrDevices;
	err = clGetDeviceIDs(intel_platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &nbrDevices);
	cl_device_id *deviceIds = (cl_device_id *)malloc(sizeof(cl_device_id) * nbrDevices);
	err = clGetDeviceIDs(intel_platform_id, CL_DEVICE_TYPE_ALL, nbrDevices, deviceIds, NULL);
	for (int i = 0; i < nbrDevices; i++)
	{
		char deviceName[128];
		cl_device_type deviceType;
		err = clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, 128, deviceName, NULL);
		err = clGetDeviceInfo(deviceIds[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL);
		printf("dev#%d - %s, %x\n", i, deviceName, deviceType);
	}
	free(deviceIds);

    cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)intel_platform_id, NULL };

    // create the OpenCL context on a CPU/PG 
	if(g_RunOnGPU)
	{
		g_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
	}
	else
	{
		g_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_CPU, NULL, NULL, &err);
	}
    if (g_context == (cl_context)0)
	{
		printf("Couldn't get context of required type, err = %d\n", err);
        return -1;
	}


    // get the list of CPU devices associated with context
    err = clGetContextInfo(g_context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    clGetContextInfo(g_context, CL_CONTEXT_DEVICES, cb, devices, NULL);

    if( alignment )
    {
        err = clGetDeviceInfo (devices[0],
            CL_DEVICE_MEM_BASE_ADDR_ALIGN ,
            sizeof(cl_uint),
            alignment,
            NULL);

        *alignment/=8; //in bytes
        printf("OpenCL data alignment is %d bytes.\n", *alignment);
    }

    g_cmd_queue = clCreateCommandQueue(g_context, devices[0], 0, &err);
    if (g_cmd_queue == (cl_command_queue)0)
    {
		printf("Unable to create command queue, err =%d\n", err);
        Cleanup_OpenCL();
        return -1;
    }

    char *sources = ReadSources(program_source);	//read program .cl source file
	if( NULL == sources )
	{
        printf("ERROR: Failed to read sources into memory...\n");
        Cleanup_OpenCL();
        return -1;
    }

    g_program = clCreateProgramWithSource(g_context, 1, (const char**)&sources, NULL, &err);
    if (g_program == (cl_program)0)
    {
        printf("ERROR: Failed to create Program with source, err = %d\n", err);
        Cleanup_OpenCL();
        free(sources);
        return -1;
    }

    err = clBuildProgram(g_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: Failed to build program, err = %d\n", err);
        BuildFailLog(g_program, devices[0]);
        Cleanup_OpenCL();
        free(sources);
        return -1;
    }

    g_kernel = clCreateKernel(g_program, "MatrixMultiply", &err);
    if (g_kernel == (cl_kernel)0)
    {
        printf("ERROR: Failed to create kernel, err= %d\n", err);
        Cleanup_OpenCL();
        free(sources);
        return -1;
    }
    free(sources);

    return 0; // success...
}

bool matrixMultiplyOpenCL(cl_float *matrixA, cl_float *matrixB, cl_float *matrixResult)
{
	cl_int err;

    //create OpenCL buffer using input array memory
    cl_mem matrixABuffer = clCreateBuffer(g_context, CL_MEM_USE_HOST_PTR, sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, matrixA, &err);

    if (matrixABuffer == (cl_mem)0)
    {
        printf("ERROR: Failed to create matrixA Buffer, err = %d\n", err);
        return false;
    }

    //create OpenCL buffer using input array memory
    cl_mem matrixBBuffer = clCreateBuffer(g_context, CL_MEM_USE_HOST_PTR, sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, matrixB, &err);

    if (matrixBBuffer == (cl_mem)0)
    {
        printf("ERROR: Failed to create matrixB Buffer, err = %d\n", err);
        return false;
    }

	cl_mem matrixResultBuffer = clCreateBuffer(g_context, CL_MEM_USE_HOST_PTR, sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, matrixResult, &err);
    if (matrixResultBuffer == (cl_mem)0)
    {
        printf("ERROR: Failed to create matrix result Buffer, err = %d\n", err);
        return false;
    }

	cl_int matrixSize = ARRAY_SIZE;

	err  = clSetKernelArg(g_kernel, 0, sizeof(cl_mem), (void *) &matrixABuffer);
    err |= clSetKernelArg(g_kernel, 1, sizeof(cl_mem), (void *) &matrixBBuffer);
    err |= clSetKernelArg(g_kernel, 2, sizeof(cl_mem), (void *) &matrixResultBuffer);
    err |= clSetKernelArg(g_kernel, 3, sizeof(cl_uint), (void *) &matrixSize );
    if (err != CL_SUCCESS)
    {
        printf("ERROR: Failed to set input kernel arguments\n");
        return false;
    }

	size_t global_work_size[2] = {matrixSize, matrixSize};

	// execute kernel
    if (CL_SUCCESS != clEnqueueNDRangeKernel(g_cmd_queue, g_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL))
    {
        printf("ERROR: Failed to execute sorting kernel\n");
        return false;
    }

	if (clFinish(g_cmd_queue) != CL_SUCCESS)
	{
		printf("Finish failed\n");
		return false;
	}

	if ((err = clEnqueueReadBuffer(g_cmd_queue, matrixResultBuffer, CL_TRUE, 0, sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, matrixResult, 0, NULL, NULL)) != CL_SUCCESS)
	{
		printf("Enqueue ReadBuffer failed: %d\n", err);
		return false;
	}

	if ((err = clReleaseMemObject(matrixABuffer)) != CL_SUCCESS)
	{
		printf("Release of matrix A buffer failed: %d", err);
		return false;
	}

	if ((err = clReleaseMemObject(matrixBBuffer)) != CL_SUCCESS)
	{
		printf("Release of matrix B buffer failed: %d", err);
		return false;
	}

	if ((err = clReleaseMemObject(matrixResultBuffer)) != CL_SUCCESS)
	{
		printf("Release of matrix result buffer failed: %d", err);
		return false;
	}

	return true;
}

void VerifyResults(cl_float *matrixResultBase, cl_float *matrixResultTest)
{
	int errorCount = 0;
	cl_float biggestAbsDiff = 0.0;
	cl_float biggestError = 0.0;
	cl_float *nextBase = matrixResultBase;
	cl_float *nextTest = matrixResultTest;
	for (int i = 0; i < ARRAY_SIZE; i++)
		for (int j = 0; j < ARRAY_SIZE; j++)
		{
			cl_float absDiff = fabs(*nextBase - *nextTest);
			if (absDiff > biggestAbsDiff)
				biggestAbsDiff = absDiff;

			cl_float relDiff = absDiff / *nextBase;
			if (relDiff > 1.0e-5)
			{
				if (relDiff > biggestError)
					biggestError = relDiff;
				errorCount++;
			}
			nextBase++;
			nextTest++;
		}

	if (biggestAbsDiff > 0.0)
		printf("Biggest abs difference = %f\n", biggestAbsDiff);
	if (errorCount == 0)
		printf("Results match\n");
	else
		printf("%d differences in results, biggest error = %f\n", errorCount, biggestError);
}

//Added by Dustin for debugging
//Print array
void print_array(cl_float* arr, char c){
	printf("Matrix%c\n",c);
	for(int i=0;i<ARRAY_SIZE;i++){
		for(int j=0;j<ARRAY_SIZE;j++){
			printf("%.1f ",*(arr + (i*ARRAY_SIZE + j)));
		}
		printf("\n");
	}
	printf("\n");
}

int _tmain(int argc, _TCHAR* argv[])
{
	int maxThreads = 4;
    srand(12345);

	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-c") == 0)
			g_RunOnGPU = false;
		else if (strncmp(argv[i], "-n:", 3) == 0)
			g_wantedName = &argv[i][3];
		else if (strncmp(argv[i], "-t:", 3) == 0)
			maxThreads = atoi(&argv[i][3]);
	}

    printf("Input size is %d items\n", ARRAY_SIZE);
    cl_float* matrixA = (cl_float*)_aligned_malloc(sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, 4);
	generateInput(matrixA);
    cl_float* matrixB = (cl_float*)_aligned_malloc(sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, 4);
	generateInput(matrixB);

    cl_float* matrixResultBase = (cl_float*)_aligned_malloc(sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, 4);
	
    QueryPerformanceFrequency(&g_PerfFrequency);
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
	matrixMultiply(matrixA, matrixB, matrixResultBase);
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);
    printf("Direct matrix multiply = %f ms.\n", 
        1000.0f*(float)(g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart);

    cl_float* matrixResultThreads = (cl_float*)_aligned_malloc(sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, 4);
	for (int nbrThreads = 1; nbrThreads <= maxThreads; nbrThreads++)
	{
		QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
		matrixMultiplyWithThreads(nbrThreads, matrixA, matrixB, matrixResultThreads);
		QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);
	    printf("Threaded matrix multiply, %d threads = %f ms.\n", nbrThreads,
			1000.0f*(float)(g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart);
		VerifyResults(matrixResultBase, matrixResultThreads);
	}
	
	if (Setup_OpenCL("MatrixMultiply.cl", NULL) != 0)
	{
		Cleanup_OpenCL();
		return 1;
	}

	cl_float* matrixResultOpenCL = (cl_float*)_aligned_malloc(sizeof(cl_float) * ARRAY_SIZE * ARRAY_SIZE, 4);

    QueryPerformanceFrequency(&g_PerfFrequency);
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
	matrixMultiplyOpenCL(matrixA, matrixB, matrixResultOpenCL);
	QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);
    printf("OpenCL matrix multiply = %f ms.\n", 
        1000.0f*(float)(g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart);
	VerifyResults(matrixResultBase, matrixResultOpenCL);
	Cleanup_OpenCL();


	printf("Press any key to end:");
	getchar();

	return 0;
}