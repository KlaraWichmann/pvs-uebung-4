// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "CL/cl.h"                              // including Open-CL header
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_SIZE   10                          // size of data / array with 10 values
#define MEM_SIZE    DATA_SIZE * sizeof(float)   // size of needed memory to be allocated for data (values are of type float) - in byte

/** source code of kernel as string **/ 
const char *KernelSource =
	"#define DATA_SIZE 10						       \n"
	"__kernel void test(__global float *input, __global float *output)   \n"
	"{									\n"
	"	size_t i = get_global_id(0);					\n"
	"	output[i] = input[i] * input[i];				\n"
	"}									\n"
	"									\n";

/** main program **/
int main (void)
{
	cl_int				err;                      	// used for making sure that request caused no errors (error status)
	cl_platform_id*		platforms = NULL;         	// number which defines the platform (e.g. intel-platform/IDE, ...)
	char			    	platform_name[1024];      	// name of platform with max 1024 characters
	cl_device_id	    		device_id = NULL;         	// number which defines the device
	cl_uint			num_of_platforms = 0,      	// define variable for total number of platforms 
					num_of_devices = 0;        	// define variable for total number of devices 
	cl_context 			context;                  	// context (needed for creating command queue)
	cl_kernel 			kernel;                   	// kernel (central element of operating system)
	cl_command_queue		command_queue;            	// command queue (queue containing the commands)
	cl_program 			program;                  	// program (created from kernel source code)
	cl_mem				input, output;            	// buffer for input + output
	float				data[DATA_SIZE] =         	// create array of size DATA_SIZE containing values of type float (numbers 1-10)
					{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	size_t				global[1] = {DATA_SIZE};  	// create array of size 1 containing value of DATA_SIZE
	float				results[DATA_SIZE] = {0}; 	// create array of size DATA_SIZE containing values of type float (number 0)

	/* 1) */

	// write total number of platforms in num_of_platforms, give return statement to err to test if platforms were found
	err = clGetPlatformIDs(0, NULL, &num_of_platforms);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}

	// reserve num_of_platforms memory
	// want to get num_of_platforms platform IDs, give return statement to err to test if platforms were found
	platforms = (cl_platform_id *)malloc(num_of_platforms);
	err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("No platforms found. Error: %d\n", err);
		return 0;
	}
	else
	{
		int nvidia_platform = 0;

		// for loop going over all platforms in 'platforms'
		for (unsigned int i=0; i<num_of_platforms; i++)
		{
			// get the name of the current platform
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Could not get information about platform. Error: %d\n", err);
				return 0;
			}
			
			// if the nvidia platform is found by name, assign it to the variable nvidia_platform
			if (strstr(platform_name, "NVIDIA") != NULL)
			{
				nvidia_platform = i;
				break;
			}
		}

		// paramter of clGetDeviceIDs: relevant platform, what device we want(here: graphic card), we want the 1st graphic card, device_id is 			saved in device_id, total number of available graphic cards in num_of devices
		//assign return statement to variable err to test if device was found
		err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
		if (err != CL_SUCCESS)
		{
			printf("Could not get device in platform. Error: %d\n", err);
			return 0;
		}
	}

	// given the device_id, a context is returned which is needed to create a command queue
	// parameter: (properties/platform, num_devices, devices, callback, user_data, errorcode) 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error: %d\n", err);
		return 0;
	}

	// given the context and device_id a command queue is returned
	// parameter: (context, device, properties/platform, errorcode)
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error: %d\n", err);
		return 0;
	}

	// creates (online) program from sourcecode of kernel as string
	// parameters: (context, count, kernel as string, length of string/ NULL if terminated, errorcode)
	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program. Error: %d\n", err);
		return 0;
	}

  	// translates program (binary code for GPU)
  	// parameters: (program, num_devices, device_list, options, callback, user_data)
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error building program. Error: %d\n", err);
		return 0;
	}

	// transform program to kernel
	// program has multiple functions and starting point must be defined (main)
	// now as starting point test and having a kernel which is able to run on the GPU
	// parameters: (program, kernel_name, errorcode)
	kernel = clCreateKernel(program, "test", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error setting kernel. Error: %d\n", err);
		return 0;
	}


	/* 2) */

	// create buffer for in and output
	// parameter: (context, flag (e.g. CL_MEM_READ_WRITE, CL_MEM_READ_ONLY, ..), size (in byte), host pointer, errorcodde)
	input  = clCreateBuffer (context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
	output = clCreateBuffer (context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, &err);

	// copy related data from array 'data' in buffer for input 'input'
	// parameter: (command_queue, buffer, blocking_write, offset, size (in_byte), data pointer, num_events_in_waiting_list, event_waiting_list, 		// event)
	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, MEM_SIZE, data, 0, NULL, NULL);

	// define order of arguments of kernel: test(input, output)
	// parameter: (kernel, arg_index, arg_size, arg_value)
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);


	/* 3)  */

	// queueing of kernel in command queue and seperating into sections 
	// parameter: (command_queue, kernel, work dimension, global_work_offset, global_work_size (in this case 10), local_work_size, 		     	     // num_events_in_waiting_list, event_waiting_list, event)
	clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

	// waiting for the end of the operation (all commands in command queue are finished)
	clFinish(command_queue);

	// copy result from output buffer 'output' in array 'results'
	// parameter: (command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_waiting_list, event_waiting_list, event)
	clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, MEM_SIZE, results, 0, NULL, NULL);

  	// print values in array 'result' - squared values
  	for (unsigned int i=0; i < DATA_SIZE; i++)
    	printf("%f\n", results[i]);


	/* 4) */
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return 0;
}
