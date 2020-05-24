// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>

#include <vector>
#include <string>

using namespace std;

#pragma comment(lib,"OpenCL.lib")


const char* source = "__kernel void dp_mul(__global const float* A, __global const float* B, __global float* C) \n\
{ \n \
    int id = get_global_id(0); \n \
    C[id] = A[id] + B[id]; \n \
    for (id = 0; id < 100000; ++id) { C[id] = A[id] + B[id]; } \n \
}\n";

//for (id = 0; id < 10000000; ++id) { C[id] = A[id] + B[id]; }\

int getDeviceInfo(cl_device_id* devices)
{
   

    char* value;
    size_t      valueSize;
    size_t      maxWorkItemPerGroup;
    cl_uint     maxComputeUnits = 0;
    cl_ulong    maxGlobalMemSize = 0;
    cl_ulong    maxConstantBufferSize = 0;
    cl_ulong    maxLocalMemSize = 0;

    ///print the device name
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*)malloc(valueSize);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device Name: %s\n", value);
    free(value);
    

    /// print parallel compute units(CU)
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    printf("Parallel compute units: %u\n", maxComputeUnits);

    ///maxWorkItemPerGroup
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkItemPerGroup), &maxWorkItemPerGroup, NULL);
    printf("maxWorkItemPerGroup: %zd\n", maxWorkItemPerGroup);

    /// print maxGlobalMemSize
    clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxGlobalMemSize), &maxGlobalMemSize, NULL);
    printf("maxGlobalMemSize: %lu(MB)\n", maxGlobalMemSize / 1024 / 1024);

    /// print maxConstantBufferSize
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
    printf("maxConstantBufferSize: %lu(KB)\n", maxConstantBufferSize / 1024);

    /// print maxLocalMemSize
    clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxLocalMemSize), &maxLocalMemSize, NULL);
    printf("maxLocalMemSize: %lu(KB)\n", maxLocalMemSize / 1024);
    return 0;
}
#define ITEM_NUM (1024*10000)

int initMem(cl_context context, cl_kernel kernel, cl_mem * clbuf1, cl_mem* clbuf2, cl_mem* clbuf3)
{
    static cl_float buf1[ITEM_NUM];
    static cl_float buf2[ITEM_NUM];
    int i;
    int status;
    for (i = 0; i < ITEM_NUM; ++i)
    {
        buf1[i] = i*1.0;
        buf2[i] = ITEM_NUM - i*1.0;
    }

    *clbuf1 = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
         ITEM_NUM * sizeof(cl_float), buf1,     NULL);

    *clbuf2 = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        ITEM_NUM * sizeof(cl_float), buf2, NULL);

    *clbuf3 = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY,
        ITEM_NUM * sizeof(cl_float), NULL, NULL);
    status = clSetKernelArg(kernel,          // valid kernel object     
        0,               // the specific argument index of a kernel     
        sizeof(cl_mem),  // the size of the argument data     
        clbuf1      // a pointer of data used as the argument 
    );
    status = clSetKernelArg(kernel,          // valid kernel object     
        1,               // the specific argument index of a kernel     
        sizeof(cl_mem),  // the size of the argument data     
        clbuf2      // a pointer of data used as the argument 
    );
    status = clSetKernelArg(kernel,          // valid kernel object     
        2,               // the specific argument index of a kernel     
        sizeof(cl_mem),  // the size of the argument data     
        clbuf3      // a pointer of data used as the argument 
    );
    return 0;
}

int main()
{
    std::cout << "Hello World!\n";
    cl_int status;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    

    status = clGetPlatformIDs(1, &platform, NULL);//创建平台对象
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);//创建GPU设备
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);//创建context
    printf("create context return %d\n", status);
   
    getDeviceInfo(&device);
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);//创建命令队列
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &status);//创建程序对象
    printf("create program result:%d\n", status);

    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);//编译程序对象
    printf("build program result:%d\n", status);

    cl_kernel kernel; 
    kernel = clCreateKernel(program,  // a valid program object that has been successfully built     
                    "dp_mul",  // the name of the kernel declared with __kernel     
                    &status  // error return code 
                );
    printf("create kernel result:%d\n", status);

    cl_mem clbuf1, clbuf2, clbuf3;
    initMem(context, kernel, &clbuf1, &clbuf2, &clbuf3);
    

    //执行kernel

    cl_event ev;
    size_t global_work_size = ITEM_NUM;

    printf("start to execute kernel...\n");

    clEnqueueNDRangeKernel(queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        NULL, 0, NULL, &ev);

    clFinish(queue);

    cl_float* ptr;
    ptr = (cl_float*)clEnqueueMapBuffer(queue,
        clbuf3,
        CL_TRUE,
        CL_MAP_READ,
        0,
        ITEM_NUM * sizeof(cl_float),
        0, NULL, NULL, NULL);
    //check result value
    int i;
    for (i = 0; i < 3; ++i)
    {
        printf("%f\n", ptr[ITEM_NUM - 1 -i]);
    }


    clReleaseMemObject(clbuf1);
    clReleaseMemObject(clbuf2);
    clReleaseMemObject(clbuf3);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
   


    
}
