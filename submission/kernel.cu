	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include <stdio.h>
	#include <opencv2\core.hpp>
	#include <opencv2\highgui.hpp>
	#include <ctime>
	#include <opencv\cv.h>
	#include <opencv\highgui.h>
	#include <time.h>
	#include <iostream>

	#define K 16				// K is the size of block KxK
	#define W 1					// size of filter is (2W+1)x(2W+1)
	using namespace cv;
	using namespace std;




__global__ void gpuKernel(unsigned char * dst, const unsigned char *src ,int width, int height )  
// this kernel uses the global memory without shared memory, i used width and height as two parameters for juding whether 
// the pixels are inside the image size. 
{

int index;			// pixel location in global memory
	//int neighbor;			// neighbor location in global memory  // i did not use this parameter. 
		
							

	// TODO: define global X and global Y and use these to calculate global offset
	// Pixel X and Y in global memory
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
							

	// TODO: if global X and global Y are within the image proceed
	if (0<=X <width && 0<=Y < height);

	// TODO: run the filter 3x3 with two nested for loops
	float sum = 0;
										
	index = X + Y*blockDim.x * gridDim.x;

	

	for (int j = -W; j <= W; j++)			// W is the filter half width
				{
					for (int i = -W; i <= W; i++)		// W is the filter half length
					{
						// TODO: calculate X and Y for neighboring pixel
							
												
						// TODO: if neighbor X and Y are not outside of the image boundary do the calculation of neighbor pixel offset
													
						if ((Y + W <= height) && (Y - W >= 0) && (X + W <= width) && (X - W >= 0))
							// TODO: calculate the filtered value

						{
							sum = sum + src[(X + i) + (Y + j)*blockDim.x * gridDim.x]; // calculate the sum of 9 pixels
							dst[index] = sum / ((2 * W + 1) * (2 * W + 1));
						}
						else {
						
							dst[index] = src[index];  //I wanted to directly copy the rest of the input image to the output image, so I can get rid of the black line, 
							// but not successful, I don't how to fix this. 
						}
					}
				}
								
										
		//dst[index] = sum / ((2 * W + 1) * (2 * W + 1));   // calculate average of the 9 values and put into the output image. 
}
	
	
	__global__ void gpuKernelTiled(unsigned char * dst, const unsigned char * src, int width, int height)  // This is the shared memory version 
	{
		__shared__ unsigned char Tile[K][K];
		// TODO: Declare Tile as shared memory

		int lx, ly;		// lx and ly are location in shared memory
						// TODO: define lx and ly
		lx = threadIdx.x;
		ly = threadIdx.y;
		int X, Y, index = 0;	
					
		// X and Y are location of pixel in global memory and index is actual pixel location in global memory
		X = threadIdx.x + blockIdx.x * blockDim.x;
		Y = threadIdx.y + blockIdx.y * blockDim.y;
		index = X + Y*blockDim.x * gridDim.x;
		// TODO: Read from global memory and put in shared memory
		Tile[lx][ly] = src[index];    
		__syncthreads();
		// TODO: fill shared memory
		

		float sum = 0;		// sum is the filtered value that you will calculate

							// TODO: run your for loops for the filtered values
		for (int j = -W; j <= W; j++)			// W is the filter half width
		{
			for (int i = -W; i <= W; i++)		// W is the filter half length
			{
				int tmpx, tmpy;
				tmpx = lx + i;
				tmpy = ly + j;

				if ((tmpx>=0)&&(tmpx<K)&&(tmpy>=0)&&(tmpy<K))  // if the pixels are within the block
				{
					sum += Tile[tmpx][tmpy];
					dst[index] = sum / ((2 * W + 1) * (2 * W + 1));
				}
				else 
				{
					if ((Y + W <= height) && (Y - W >= 0) && (X + W <= width) && (X - W >= 0))
						// TODO: calculate the filtered value

					{
						sum = sum + src[(X + i) + (Y + j)*blockDim.x * gridDim.x]; // calculate the sum of 9 pixels
						dst[index] = sum / ((2 * W + 1) * (2 * W + 1));
					}
					else {

						dst[index] = src[index];  //I wanted to  directly copy the rest of the input image to the output image, so I can get rid of the black line, 
												  // but not successful, I don't how to fix this. 
					}
				}
							
														
			}
		}
					
			//dst[index] = sum / ((2 * W + 1) * (2 * W + 1)); // Here the filtered value will be stored in the output

	}


	// CPU 
	void cpuFilter(Mat &, const Mat&);


	// GPU helper code
	cudaError_t thresholdWithCudaNoShared(Mat&, const Mat&);
	cudaError_t thresholdWithCudaWithShared(Mat&, const Mat&);

	int main()
	{

		cudaError_t cudaStatus;
		clock_t tStart;
		int ch = 0;
		//Mat inputImage = imread("C:\\Users\\sw5\\Desktop\\course books\\6398\\Home works\\Project 2\\LightHouse_gray.jpg",0);
		//Mat inputImage = imread("C:\\Users\\sw5\\Desktop\\course books\\6398\\Home works\\Project 2\\Hydrangeas_gray.jpg",0);

		Mat inputImage = imread("C:\\Users\\sw5\\Desktop\\course books\\6398\\Home works\\Project 2\\Desert_gray.jpg", 0);
		Mat cpuTHImage(inputImage.rows, inputImage.cols, CV_8UC1);
		Mat gpuTHImage(inputImage.rows, inputImage.cols, CV_8UC1);
				
				

		if (!inputImage.data)
		{
			printf("Image didn't load properly!\n");
		}
		else
		{
			cout << "Enter 1 for CPU \n";
			cout << "Enter 2 for GPU \n";
			cout << "Enter 3 for CPU + GPU \n";
			//cout << "Enter 0 to Exit \n";
			cin >> ch;

			switch (ch)
			{
			case 1:
				// Calling CPU function
				tStart = clock(); //Starting clock
				cpuFilter(cpuTHImage, inputImage);
				printf("Time taken: %.2fms\n", (double)(clock() - tStart) / (CLOCKS_PER_SEC / 1000)); //Stopping and displaying time
																										//Displaying Input and Output Images
				imshow("Input_Image", inputImage);
				imshow("CPU_Output_Image", cpuTHImage);
				break;
			case 2:
				// Calling GPU fucntion
				tStart = clock(); //Starting clock
				thresholdWithCudaNoShared(gpuTHImage, inputImage);
				printf("Time taken: %.2fms\n", (double)(clock() - tStart)/(CLOCKS_PER_SEC/1000)); //Stopping and displaying time
				//Displaying Input and Output Images
				imshow("Input_Image", inputImage);
				imshow("GPU_Output_Image", gpuTHImage);
				break;
			case 3:
				//Calling CPU and GPU function
				tStart = clock();
				cpuFilter(cpuTHImage, inputImage);
				printf("Time taken (CPU): %.2fms\n", (double)(clock() - tStart) / (CLOCKS_PER_SEC / 1000)); //Stopping and displaying time
				tStart = clock();
				thresholdWithCudaWithShared(gpuTHImage, inputImage);
				printf("Time taken (GPU): %.2fms\n", (double)(clock() - tStart)/(CLOCKS_PER_SEC/1000)); //Stopping and displaying time
				//Displaying Input and Output Images
				imshow("Input_Image", inputImage);
				imshow("CPU_Output_Image", cpuTHImage);
				imshow("SharedGPU_Output_Image", gpuTHImage);
				break;
			default:
				break;
			}
	}

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
		cvWaitKey(0);
		return 0;
	}

	//CPU Implemenation Code
	void cpuFilter(Mat& dest, const Mat& src)
	{

	int rows = src.rows;
	int cols = src.cols;
	int sum;

	// method to deal with edges of the image, bascially use the edge of the original image or the new image. 
	for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
			dest.data[j + i*cols] = src.data[j + i*cols];
			}
		}

	for (int i = 1; i < rows-1; i++)
		{
			for (int j = 1; j < cols-1; j++)
		{
			if ((0 < i < rows - 1) && (0 < j < cols - 1))

				{
					sum = 0;
					for (int x = -W; x <= W; x++)
					{
						for (int y = -W; y <= W; y++)
							{
						sum = sum + src.data[(j + x) + (i + y)*cols];
							}
					}
						int outindex = j + i*cols;
						dest.data[outindex] = sum / ((2 * W + 1)*(2 * W + 1));
		}
				
	}
}

}
	
							
	// Helper function for using CUDA to add vectors in parallel.
	//******************************************************************************************************************************************
	// Helper function for using CUDA to perform image thresholding in parallel. Takes as input the thresholded image (bwImage), the input image (input), and the threshold value.
	cudaError_t thresholdWithCudaNoShared(Mat & outputImg, const Mat & inputImg)
	{

		// Allocate GPU buffers for the buffers (one input, one output)   
		unsigned char *dev_dst = 0;
		unsigned char *dev_src = 0;	// these are the gpu side ouput and input pointers
		int width = inputImg.size().width;
		int height = inputImg.size().height;

		cudaError_t cudaStatus;

		cudaEvent_t start, stop;	// These are your start and stop events to calculate your GPU performance
		float time = 0;                                     // This is the gpu time	

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}
				

		// TODO: add your code here to allocate the input pointer on the device. Note the size of the pointer in cudaMalloc
		cudaStatus = cudaMalloc((void**)& dev_src, sizeof(unsigned char)*inputImg.rows*inputImg.cols);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Cuda failed");
			goto Error;
		}

		// TODO: add your code here to allocate the output pointer on the device. Note the size of the pointer in cudaMalloc
		cudaStatus = cudaMalloc((void**)& dev_dst, sizeof(unsigned char)*outputImg.rows*outputImg.cols);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Cuda failed");
			goto Error;
		}
				
				
			
		// Copy input data from host memory to GPU buffers.
		// TODO: Add your code here. Use cudaMemcpy
		cudaStatus = cudaMemcpy(dev_src, inputImg.data, sizeof(unsigned char)*inputImg.rows*inputImg.cols, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Cuda failed");
			goto Error;
		}

		// TODO: Launch a kernel on the GPU with one thread for each element. use <<< grid_size (or number of blocks), block_size(or number of threads) >>>
		dim3 block(K, K, 1);
		dim3 grid(inputImg.cols / K, inputImg.rows / K, 1);

		// lauch a kernel on the GPU with one thread for each element.
					
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		gpuKernel <<<grid, block >>> (dev_dst, dev_src, width, height);
		// TODO: record your stop event on GPU
		cudaEventRecord(stop);
		// TODO: Synchronize stop event
		cudaEventSynchronize(stop);
		// TODO: calculate the time ellaped on GPU
					
		cudaEventElapsedTime(&time, start, stop);
		printf("Global Memory time=%3.2f ms\n", time);

			
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
					
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		// TODO: Copy output data from GPU buffer to host memory. use cudaMemcpy
		cudaStatus = cudaMemcpy(outputImg.data, dev_dst, sizeof(unsigned char)*outputImg.rows*outputImg.cols, cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Cuda failed");
			goto Error;
		}
	Error:
		cudaFree(dev_src);
		cudaFree(dev_dst);

		return cudaStatus;
	}

	//******************************************************************************************************
	cudaError_t thresholdWithCudaWithShared(Mat & destImg, const Mat & srcImg)
	{
		unsigned char *dev_src = 0;
		unsigned char *dev_dst = 0;
		int width = srcImg.size().width;
		int height = srcImg.size().height;

		cudaError_t cudaStatus;		// cuda status variable for errors on GPU

		cudaEvent_t start, stop;	// These are your start and stop events to calculate your GPU performance
		float time = 0;					// This is the gpu time


										// TODO: register your events for GPU 


										// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		// Allocate GPU buffers for two vectors (One input, one output)   

		cudaStatus = cudaMalloc((void**)& dev_src, sizeof(unsigned char) * srcImg.rows * srcImg.cols);


		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		//target image
		cudaStatus = cudaMalloc((void **)& dev_dst, sizeof(unsigned char) * destImg.rows * destImg.cols);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}



		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_src, srcImg.data, sizeof(unsigned char) * srcImg.rows * srcImg.cols, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy (CPU ->GPU) failed!");
			goto Error;
		}


		// Launch a kernel on the GPU with one thread for each element.

		dim3 block(K, K, 1);
		dim3 grid(srcImg.cols / K, srcImg.rows / K, 1);



		// TODO: record your start event on GPU
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		gpuKernelTiled <<<grid, block >> >(dev_dst, dev_src, width, height);		// invking the kernel with tiled shared memory
																					// TODO: record your stop event on GPU
		cudaEventRecord(stop);
		// TODO: Synchronize stop event
		cudaEventSynchronize(stop);
		// TODO: calculate the time ellaped on GPU
					
		cudaEventElapsedTime(&time, start, stop);
		printf("Shared Memory time=%3.2f ms\n", time);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(destImg.data, dev_dst, sizeof(unsigned char) * destImg.rows * destImg.cols, cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy (GPU -> CPU) failed!");
			goto Error;
		}

	Error:
		cudaFree(dev_src);
		cudaFree(dev_dst);
			
		return cudaStatus;
	}



