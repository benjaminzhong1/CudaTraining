
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <vector>
#include <random>

__global__ void vector_multiply_single(int* A, int* B, int* C, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int temp_sum = 0;

	for (int x = 0; x < width; x++)
	{
		int a = A[tx * width + x];
		int b = B[x * width + ty];
		temp_sum += a * b;
	}
	
	C[tx * width + ty] += temp_sum;
}


__global__ void mat_mul_kernel(int* M, int* N, int* P, int Width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int inter_sum;

	for (int x = 0; x < Width; x++)
	{
		int m_mat_value = M[ty * Width + x];
		int n_mat_value = N[x * Width + tx];
		inter_sum += m_mat_value * n_mat_value;
	}

	P[ty * Width + tx] = inter_sum;
}

__global__ void my_kernel()
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	std::printf("Thread IDx: %d , Thread IDy: %d , BlockID: %d\n", tx, ty, bx);
}

__global__ void vector_multiply_multiple(void);

//generate array will generate a 512 length array that's filled with random numbers that range from 1-1000
int* generate_array(void)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distribution(1, 100);

	int* array = new int[256];
	for (int x(0); x < 256; x++)
	{
		array[x] = distribution(gen);
		std::printf("%d ", array[x]);
	}
	std::printf("\n");
	return array;
}

void mat_mul(int* M, int* N, int* P, int Width)
{
	for (int x = 0; x < Width; x++)
	{
		for (int y = 0; y < Width; y++)
		{
			int temp_product = 0;

			for (int z = 0; z < Width; z++)
			{
				int row_val = M[x * Width + z];
				int col_val = N[z * Width + y];
				temp_product += row_val * col_val;
			}

			P[x * Width + y] = temp_product;
		}
	}
}

int main()
{
	//generate arrays of width
	int width = 16;

	//dynamically allocated arrays
	int* A = generate_array();
	int* B = generate_array();
	//solution arrays
	int* C = new int[256];
	int* D = new int[256];


	//declare 4 pointers for mem alloc on device
	int* W, * X, * Y, * Z;
	int array_bit_length = 4 * 256;

	
	////malloc those 4 ptrs on length of bytes
	cudaMalloc(&W, array_bit_length);
	cudaMalloc(&X, array_bit_length);
	cudaMalloc(&Y, array_bit_length);
	//cudaMalloc(&Z, array_bit_length);


	////memcopy our two arrays over
	cudaMemcpy(W, A, array_bit_length, cudaMemcpyHostToDevice);
	cudaMemcpy(X, B, array_bit_length, cudaMemcpyHostToDevice);

	dim3 dimGrid(1, 1);
	dim3 dimBlock(16, 16);

	////perform opp on two arrays with vector_multiply
	vector_multiply_single << < dimGrid, dimBlock>> > (W, X, Y, width);
	
	//vector_multiply_multiple << < temp, temp >> > (A, B, D, width);

	////copy over result array back to host for single block + multiple threads
	////copy over result array back to host for multiple blocks + threads
	cudaMemcpy(C, Y, array_bit_length, cudaMemcpyDeviceToHost);
	//cudaMemcpy(D, Z, array_bit_length, cudaMemcpyDeviceToHost);

	mat_mul(A, B, D, 16);


	////cudafree for device memory
	cudaFree(W);
	cudaFree(X);
	cudaFree(Y);
	//cudaFree(Z);

	for (int x = 0; x < 256; x++)
	{
		std::printf("%d ", C[x]);
	}

	bool check = false;

	for (int x = 0; x < 256; x++)
	{
		if (C[x] == D[x])
		{
			check = true;
		}
		else
		{
			check = false;
			break;
		}
	}

	std::cout << check << std::endl;

	//memory deallocation and set ptrs to nullptr
	delete A, B, C, D;

	std::printf("hello world");
	return 0;
}





