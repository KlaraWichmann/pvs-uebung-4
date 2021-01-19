#include "CL/cl.h"                         
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAT_SIZE 1000
#define DATA_SIZE   MAT_SIZE*MAT_SIZE*sizeof(float)                         
#define MEM_SIZE    DATA_SIZE * sizeof(float)
#define EPSILON 0.0001f



float** alloc_mat(int row, int col) {
    float **A1, *A2;

    A1 = (float**)calloc(row, sizeof(float*));
    A2 = (float*)calloc(row * col, sizeof(float));
    for (int i = 0; i < row; i++)
        A1[i] = A2 + i * col;

    return A1;
}

void init_mat(float** A, int row, int col) {
    for (int i = 0; i < row * col; i++)
        A[0][i] = (float)(rand() % 10);
}


void print_mat(float** A, int row, int col, char const* tag) {
    int i, j;

    printf("Matrix %s:\n", tag);
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++)
            printf("%6.1f   ", A[i][j]);
        printf("\n");
    }
}

void free_mat(float** A, int num_rows) {
    free(A[0]);
    free(A);
}

bool mat_equal(float** mat1, float** mat2, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (abs(mat1[i][j] - mat2[i][j]) > EPSILON) {
                return false;
            }
        }
    }
    return true;
}


/* 
 const char *KernelSource =
	"#define DATA_SIZE 10						       																\n"
	"__kernel void test(__global float *input, __global float *output)   	\n"
	"{																																		\n"
	"	size_t i = get_global_id(0);																				\n"
	"	output[i] = input[i] * input[i];																		\n"
	"}																																		\n"
	"																																			\n";

 */



const char *KernelSource =
"#define MAT_SIZE 1000		// 	Groesse der Matrix 											\n"
"__kernel void matmult(__gloabl float *A, __global float *B, 		\n"
"											 __global float *C, __global float *B1) {	\n"
"				int k, j, i = get_global_id(0);													\n"
"				float A1[MAT_SIZE], sum;																			\n"
"				int i1 = get_local_id(0);																\n"
"				int n1 = get_local_size(0);															\n"
"				for (k = 0; k < MAT_SIZE; k++)																\n"
"					A1[k] = A[i*MAT_SIZE+k];																		\n"
"				for (j = 0; j < MAT_SIZE; j++) {															\n"
"					for (k = i1; k < MAT_SIZE; k+=n1)													\n"
"						B1[k] = B[k*MAT_SIZE+j];																	\n"
"					barrier(CLK_LOCAL_MEM_FENCE); 		// Warten						\n"
"					sum = 0.0;																						\n"
"					for (k = 0; k < MAT_SIZE; k++)															\n"
"						sum += A1[k] * B1[k];																\n"
"					C[i*MAT_SIZE+j] = sum;																			\n"
"				}																												\n"
"}																															\n"
"\n";

 


/** main program **/
int main (void) {
	cl_int						err; 
	cl_platform_id*		platforms = NULL;  
	char			    		platform_name[1024]; 
	cl_device_id	    device_id = NULL;         
	cl_uint						num_of_platforms = 0,     
										num_of_devices = 0;        
	cl_context 				context;                 
	cl_kernel 				kernel;                   	
	cl_command_queue	command_queue;            
	cl_program 				program;                  
	cl_mem						Ap, Bp, Cp;           
	float							data[DATA_SIZE] =         	
										{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	size_t						global[1] = {MAT_SIZE};  
	float							results[DATA_SIZE] = {0}; 


  float **A, **B, **C;

  A = alloc_mat(MAT_SIZE, MAT_SIZE); init_mat(A, MAT_SIZE, MAT_SIZE);
  B = alloc_mat(MAT_SIZE, MAT_SIZE); init_mat(B, MAT_SIZE, MAT_SIZE);
  C = alloc_mat(MAT_SIZE, MAT_SIZE);


	/* 1) */

	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS) {
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	platforms = (cl_platform_id *)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS) {
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}
	else {
		int nvidia_platform = 0;


		for (unsigned int i=0; i<num_of_platforms; i++) {
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
			if (err != CL_SUCCESS) {
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}
			
			if (strstr(platform_name, "NVIDIA") != NULL) {
				nvidia_platform = i;
				break;
			}
		}

		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS) {
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS) {
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	kernel = clCreateKernel(program, "matmult", &err);
	if (err != CL_SUCCESS) {
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}

	/* 2) */
	Ap = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_SIZE, NULL, &err);
  Bp = clCreateBuffer(context, CL_MEM_READ_ONLY, DATA_SIZE, NULL, &err);
  Cp = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DATA_SIZE, NULL, &err);
	clEnqueueWriteBuffer(command_queue, Ap, CL_TRUE, 0, DATA_SIZE, A[0], 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, Bp, CL_TRUE, 0, DATA_SIZE, B[0], 0, NULL, NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &Ap);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &Bp);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &Cp);


	/* 3)  */
	clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
	clFinish(command_queue);
	clEnqueueReadBuffer(command_queue, Cp, CL_TRUE, 0, DATA_SIZE, C[0], 0, NULL, NULL);

  for (unsigned int i=0; i < DATA_SIZE; i++)
    printf("%f\n", results[i]);


	/* 4) */
	clReleaseMemObject(Ap);
	clReleaseMemObject(Bp);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);


  free_mat(A, MAT_SIZE);
  free_mat(B, MAT_SIZE);
  free_mat(C, MAT_SIZE);


	return 0;
}
