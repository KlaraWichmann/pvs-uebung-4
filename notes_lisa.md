# Version 6: C zeilenweise, A privat, B lokal

```cpp

const char *KernelSource =
"#define DIM 1000		// 	Groesse der Matrix 											\n"
"__kernel void matmult(__gloabl float *A, __global float *B, 		\n"
"											 __global float *C, __global float *B1) {	\n"
"				int k, j, i = get_global_id(0);													\n"
"				float A1[DIM], sum;																			\n"
"				int i1 = get_local_id(0);																\n"
"				int n1 = get_local_size(0);															\n"
"				for (k = 0; k < DIM; k++)																\n"
"					A1[k] = A[i*DIM+k];																		\n"
"				for (j = 0; j < DIM; j++) {															\n"
"					for (k = i1; k < DIM; k+=n1)													\n"
"						B1[k] = B[k*DIM+j];																	\n"
"					barrier(CLK_LOCAL_MEM_FENCE); 		// Warten						\n"
"					sum = 0.0;																						\n"
"					for (k = 0; k < DIM; k++)															\n"
"						sum += A1[k] * B1[k];																\n"
"					C[i*DIM+j] = sum;																			\n"
"				}																												\n"
"}																															\n"
"\n";

```

Error log:
Error building program. Error: -11