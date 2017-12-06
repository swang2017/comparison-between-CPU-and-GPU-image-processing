#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <time.h>

#define K 16				// K is the size of block KxK
#define W 1					// size of filter is (2W+1)x(2W+1)
using namespace cv;
using namespace std;




__global__ void gpuKernel(unsigned char * dst, const unsigned char * src)  // this kernel uses the global memory without shared memory
{

	int index=0;			// pixel location in global memory
	int neighbor;			// neighbor location in global memory



	float sum=0;
	
	// TODO: define global X and global Y and use these to calculate global offset

	int X, Y;			// Pixel X and Y in global memory

	// TODO: if global X and global Y are within the image proceed


	// TODO: run the filter 3x3 with two nested for loops

	for (int j=-W;j<=W;j++)			// W is the filter half width
	{
		for (int i=-W;i<=W;i++)		// W is the filter half length
		{
			// TODO: calculate X and Y for neighboring pixel
			
			// TODO: if neighbor X and Y are not outside of the image boundary do the calculation of neighbor pixel offset

			// TODO: calculate the filtered value
		}
	}

	dst[index] = sum/(2*W+1)/(2*W+1);

}

__global__ void gpuKernelTiled(unsigned char * dst, const unsigned char * src)  // This is the shared memory version 
{
	
	// TODO: Declare Tile as shared memory

	int lx , ly;		// lx and ly are location in shared memory
	// TODO: define lx and ly

	int X, Y, index=0;	// X and Y are location of pixel in global memory and index is actual pixel location in global memory

	// TODO: Read from global memory and put in shared memory

	// TODO: fill shared memory

	float sum = 0;		// sum is the filtered value that you will calculate

	// TODO: run your for loops for the filtered values

	dst[index] = sum/(2*W+1)/(2*W+1);	// Here the filtered value will be stored in the output

}


// CPU code
void cpuFilter(Mat &, const Mat&);

// GPU helper code
cudaError_t thresholdWithCudaNoShared(Mat&, const Mat&);
cudaError_t thresholdWithCudaWithShared(Mat&, const Mat&);

int main()
{

	cudaError_t cudaStatus;
	clock_t tStart;
	int ch = 0;
	//Mat inputImage = imread("C:\\opencv\\LightHouse_gray.jpg",0);
	//Mat inputImage = imread("C:\\opencv\\Hydrangeas_gray.jpg",0);

	Mat inputImage = imread("C:\\images\\Desert_gray.jpg",0);
	Mat cpuTHImage (inputImage.rows, inputImage.cols, CV_8UC1);
	Mat gpuTHImage (inputImage.rows, inputImage.cols, CV_8UC1);

	if(!inputImage.data)
	{
		printf ("Image didn't load properly!\n");
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
			printf("Time taken: %.2fms\n", (double)(clock() - tStart)/(CLOCKS_PER_SEC/1000)); //Stopping and displaying time
			//Displaying Input and Output Images
			imshow("Input_Image",inputImage);
			imshow("CPU_Output_Image",cpuTHImage);
			break;
		case 2:
			// Calling GPU fucntion
			//tStart = clock(); //Starting clock
			thresholdWithCudaNoShared(gpuTHImage, inputImage);
			//printf("Time taken: %.2fms\n", (double)(clock() - tStart)/(CLOCKS_PER_SEC/1000)); //Stopping and displaying time
			//Displaying Input and Output Images
			imshow("Input_Image",inputImage);
			imshow("GPU_Output_Image",gpuTHImage);
			break;
		case 3:
			//Calling CPU and GPU function
			 tStart = clock(); 
			cpuFilter(cpuTHImage, inputImage);
			printf("Time taken (CPU): %.2fms\n", (double)(clock() - tStart)/(CLOCKS_PER_SEC/1000)); //Stopping and displaying time
			 tStart = clock(); 
			thresholdWithCudaWithShared(gpuTHImage, inputImage);
			//printf("Time taken (GPU): %.2fms\n", (double)(clock() - tStart)/(CLOCKS_PER_SEC/1000)); //Stopping and displaying time
			//Displaying Input and Output Images
			imshow("Input_Image",inputImage);
			imshow("NoSharedGPU_Output_Image",cpuTHImage);
			imshow("SharedGPU_Output_Image",gpuTHImage);
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

	// TODO: Write your CPU code here

}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t thresholdWithCudaNoShared(Mat & destImg, const Mat & srcImg)
{
	unsigned char *dev_src = 0;
	unsigned char *dev_dst = 0;

	cudaError_t cudaStatus;		// cuda status variable for errors on GPU

	cudaEvent_t start, stop;	// These are your start and stop events to calculate your GPU performance
	                			// This is the gpu time
	

	// TODO: register your events for GPU 
	

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (One input, one output)   

	cudaStatus = cudaMalloc( (void**) & dev_src, sizeof(unsigned char) * srcImg.rows * srcImg.cols);


	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//target image
	cudaStatus = cudaMalloc( (void **) & dev_dst, sizeof(unsigned char) * destImg.rows * destImg.cols);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}



	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy (dev_src, srcImg.data, sizeof(unsigned char) * srcImg.rows * srcImg.cols, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (CPU ->GPU) failed!");
		goto Error;
	}



	// Launch a kernel on the GPU with one thread for each element.

	dim3 block(K, K, 1);
	dim3 grid(srcImg.cols/K, srcImg.rows/K, 1);


	// TODO: record your start event on GPU
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	gpuKernel << <grid, block >> > (dev_dst, dev_src, width, height);  // invoke the kernel
	// TODO: record your stop event on GPU
	cudaEventRecord(stop);
	// TODO: Synchronize stop event
	cudaEventSynchronize(stop);
	// TODO: calculate the time ellaped on GPU
	float time = 0;
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

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(destImg.data, dev_dst, sizeof (unsigned char) * destImg.rows * destImg.cols, cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (GPU -> CPU) failed!");
		goto Error;
	}

Error:
	cudaFree(dev_src);
	cudaFree(dev_dst);

	return cudaStatus;
}

cudaError_t thresholdWithCudaWithShared(Mat & destImg, const Mat & srcImg)
{
	unsigned char *dev_src = 0;
	unsigned char *dev_dst = 0;

	cudaError_t cudaStatus;		// cuda status variable for errors on GPU

	cudaEvent_t start, stop;	// These are your start and stop events to calculate your GPU performance
	float time=0;					// This is the gpu time


	// TODO: register your events for GPU 


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (One input, one output)   

	cudaStatus = cudaMalloc( (void**) & dev_src, sizeof(unsigned char) * srcImg.rows * srcImg.cols);


	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	//target image
	cudaStatus = cudaMalloc( (void **) & dev_dst, sizeof(unsigned char) * destImg.rows * destImg.cols);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}



	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy (dev_src, srcImg.data, sizeof(unsigned char) * srcImg.rows * srcImg.cols, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (CPU ->GPU) failed!");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.

	dim3 block(K, K, 1);
	dim3 grid(srcImg.cols/K, srcImg.rows/K, 1);



	// TODO: record your start event on GPU

	gpuKernelTiled<<<grid,block>>>(dev_dst,dev_src);		// invking the kernel with tiled shared memory
	
	// TODO: record your stop event on GPU
	
	// TODO: Synchronize stop event

	// TODO: calculate the time ellaped on GPU

	printf("Global Memory time=%3.2f ms\n",time);

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
	cudaStatus = cudaMemcpy(destImg.data, dev_dst, sizeof (unsigned char) * destImg.rows * destImg.cols, cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (GPU -> CPU) failed!");
		goto Error;
	}

Error:
	cudaFree(dev_src);
	cudaFree(dev_dst);

	return cudaStatus;
}



