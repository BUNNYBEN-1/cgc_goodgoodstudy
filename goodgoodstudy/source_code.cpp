#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h> // 包含AVX指令集的头文件
#pragma GCC target("avx2")
#pragma GCC optimize("O3","unroll-loops")

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;                   //顶点数
int e_num = 0;                   //边数
int F0 = 0, F1 = 0, F2 = 0;      //F0 <= 128; F1 = 16; F2 <= 32
int num_threads = 0;             //并行执行的线程数

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

//读取文件必须使用示例文件提供的readGraph()函数，不可修改，不计入执行时间内，若需转换为邻接表或是CSR等格式须在somePreprocessing()函数内实现，并计入执行时间。
void readGraph(char *fname)
{
	ifstream infile(fname);

	int source;     //源节点
	int end;        //目的节点

	infile >> v_num >> e_num;     //文件第一行是顶点数和边数

	raw_graph.resize(e_num * 2);  //改变原图的容量为边数的二倍

	while (!infile.eof())
	{
		infile >> source >> end;       //读文件中的第二行到最后一行，每行有两个数，依次是源节点id和目的节点id
		if (infile.peek() == EOF)
			break;
		raw_graph.push_back(source);   //把源节点id添加到原图向量的末尾
		raw_graph.push_back(end);      //把目的节点id添加到原图向量的末尾
	}
}

//将原图向量转为邻接矩阵
void raw_graph_to_AdjacencyList()
{
	int src;
	int dst;

	edge_index.resize(v_num);
	edge_val.resize(v_num);
	degree.resize(v_num, 0);

	for (int i = 0; i < raw_graph.size() / 2; i++)
	{
		src = raw_graph[2 * i];
		dst = raw_graph[2 * i + 1];
		edge_index[dst].push_back(src);
		degree[src]++;
	}
}

//边的归一化
void edgeNormalization()
{
	for (int i = 0; i < v_num; i++)
	{
		for (int j = 0; j < edge_index[i].size(); j++)
		{
			float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
			edge_val[i].push_back(val);
		}
	}
}

/*
void readFloat(char *fname, float *&dst, int num)
{
	dst = (float *)malloc(num * sizeof(float));
	FILE *fp = fopen(fname, "rb");
	fread(dst, num * sizeof(float), 1, fp);
	fclose(fp);
}*/

void readFloat(const char* fname, float*& dst, int num)
{
	dst = (float*)malloc(num * sizeof(float));

	FILE* fp = fopen(fname, "rb");
	if (fp == nullptr)
	{
		// 文件打开失败，进行错误处理
		printf("无法打开文件：%s\n", fname);
		// 其他错误处理代码
		// ...
	}
	else
	{
		// 文件成功打开，可以进行文件读取操作
		// 读取文件内容
		fread(dst, sizeof(float), num, fp);
		fclose(fp);
	}
}

void initFloat(float *&dst, int num)
{
	dst = (float *)malloc(num * sizeof(float));
	memset(dst, 0, num * sizeof(float));
}

/*多线程
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
	omp_set_num_threads(num_threads);

#pragma omp parallel for
	for (int i = 0; i < v_num; i++)
	{
		for (int j = 0; j < out_dim; j++)
		{
			for (int k = 0; k < in_dim; k++)
			{
				out_X[i * out_dim + j] += in_X[i * in_dim + k] * W[k * out_dim + j];
			}
		}
	}
}*/

/*AVX指令集
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
	// 每次处理8个单精度浮点数
	for (int i = 0; i < v_num; i++)
	{
		for (int j = 0; j < out_dim; j += 8)
		{
			__m256 sum = _mm256_setzero_ps(); // 初始化累加和为零向量

			for (int k = 0; k < in_dim; k++)
			{
				// 加载输入矩阵和权重矩阵的向量
				__m256 X_vec = _mm256_loadu_ps(&in_X[i * in_dim + k]);
				__m256 W_vec = _mm256_loadu_ps(&W[k * out_dim + j]);

				// 执行乘法运算
				__m256 mul = _mm256_mul_ps(X_vec, W_vec);

				// 累加乘积结果
				sum = _mm256_add_ps(sum, mul);
			}

			// 存储累加和到输出矩阵
			_mm256_storeu_ps(&out_X[i * out_dim + j], sum);
		}
	}
}*/

/*多线程+AVX指令集*/
void XW(int in_dim, int out_dim, float* in_X, float* out_X, float* W)
{
	omp_set_num_threads(num_threads);

#pragma omp parallel for
	for (int i = 0; i < v_num; i++)
	{
		for (int j = 0; j < out_dim; j += 8)
		{
			__m256 sum = _mm256_setzero_ps(); // 初始化累加和为零向量

			for (int k = 0; k < in_dim; k++)
			{
				__m256 X_vec = _mm256_loadu_ps(&in_X[i * in_dim + k]);
				__m256 W_vec = _mm256_loadu_ps(&W[k * out_dim + j]);

				__m256 mul = _mm256_mul_ps(X_vec, W_vec);

				sum = _mm256_add_ps(sum, mul);
			}

			// 存储累加和到输出矩阵
			_mm256_storeu_ps(&out_X[i * out_dim + j], sum);
		}
	}
}


/*多线程
void AX(int dim, float *in_X, float *out_X)
{
	omp_set_num_threads(num_threads);

#pragma omp parallel for
	for (int i = 0; i < v_num; i++)
	{
		std::vector<int> &nlist = edge_index[i];
		for (int j = 0; j < nlist.size(); j++)
		{
			int nbr = nlist[j];
			for (int k = 0; k < dim; k++)
			{
#pragma omp atomic
				out_X[i * dim + k] += in_X[nbr * dim + k] * edge_val[i][j];
			}
		}
	}
}*/

/*AVX指令集
void AX(int dim, float *in_X, float *out_X)
{
	for (int i = 0; i < v_num; i++)
	{
		std::vector<int> &nlist = edge_index[i];
		for (int j = 0; j < nlist.size(); j += 8) // 假设使用AVX-256，一次处理8个元素
		{
			int nbr[8];
			for (int k = 0; k < 8; k++)
			{
				if (j + k < nlist.size())
					nbr[k] = nlist[j + k];
				else
					nbr[k] = -1; // 用于处理边界情况，如果邻居索引不足8个，则填充-1
			}

			for (int k = 0; k < dim; k += 8) // 一次处理8个维度
			{
				__m256 in_vec = _mm256_loadu_ps(&in_X[i * dim + k]); // 加载输入向量中的数据
				__m256 out_vec = _mm256_loadu_ps(&out_X[i * dim + k]); // 加载输出向量中的数据

				for (int l = 0; l < 8; l++)
				{
					if (nbr[l] != -1)
					{
						__m256 edge_val_vec = _mm256_set1_ps(edge_val[i][j + l]); // 使用边界矩阵中的值创建一个向量
						__m256 in_X_nbr_vec = _mm256_loadu_ps(&in_X[nbr[l] * dim + k]); // 加载邻居节点的输入向量数据

						__m256 result_vec = _mm256_mul_ps(in_X_nbr_vec, edge_val_vec); // 对应元素相乘
						out_vec = _mm256_add_ps(out_vec, result_vec); // 累加到输出向量中
					}
				}

				_mm256_storeu_ps(&out_X[i * dim + k], out_vec); // 存储更新后的输出向量数据
			}
		}
	}
}*/

/*多线程+AVX指令集*/
void AX(int dim, float* in_X, float* out_X)
{
	omp_set_num_threads(num_threads);

#pragma omp parallel for
	for (int i = 0; i < v_num; i++)
	{
		std::vector<int>& nlist = edge_index[i];
		for (int j = 0; j < nlist.size(); j += 8)
		{
			int nbr[8];
			for (int k = 0; k < 8; k++)
			{
				if (j + k < nlist.size())
					nbr[k] = nlist[j + k];
				else
					nbr[k] = -1;
			}

			for (int k = 0; k < dim; k += 8)
			{
				__m256 in_vec = _mm256_loadu_ps(&in_X[i * dim + k]);
				__m256 out_vec = _mm256_loadu_ps(&out_X[i * dim + k]);

				for (int l = 0; l < 8; l++)
				{
					if (nbr[l] != -1)
					{
						__m256 edge_val_vec = _mm256_set1_ps(edge_val[i][j + l]);
						__m256 in_X_nbr_vec = _mm256_loadu_ps(&in_X[nbr[l] * dim + k]);

						__m256 result_vec = _mm256_mul_ps(in_X_nbr_vec, edge_val_vec);
						out_vec = _mm256_add_ps(out_vec, result_vec);
					}
				}

				_mm256_storeu_ps(&out_X[i * dim + k], out_vec);
			}
		}
	}
}

void ReLU(int dim, float *X)
{
	for (int i = 0; i < v_num * dim; i++)
		if (X[i] < 0)
			X[i] = 0;
}

void LogSoftmax(int dim, float *X)
{
	for (int i = 0; i < v_num; i++)
	{
		float max = X[i * dim];
		for (int j = 1; j < dim; j++)
		{
			if (X[i * dim + j] > max)
				max = X[i * dim + j];
		}

		float sum = 0;
		for (int j = 0; j < dim; j++)
		{
			sum += std::exp(X[i * dim + j] - max);
		}
		sum = std::log(sum);

		for (int j = 0; j < dim; j++)
		{
			X[i * dim + j] = X[i * dim + j] - max - sum;
		}
	}
}

float MaxRowSum(float *X, int dim)
{
	float max = -__FLT_MAX__;
	for (int i = 0; i < v_num; i++)
	{
		float sum = 0;
		for (int j = 0; j < dim; j++)
		{
			sum += X[i * dim + j];
		}
		if (sum > max)
			max = sum;
	}
	return max;
}

void freeFloats()
{
	free(X0);
	free(W1);
	free(W2);
	free(X1);
	free(X2);
	free(X1_inter);
	free(X2_inter);
}

void somePreprocessing()
{
	//The graph  will be transformed into adjacency list ,you can use other data structure such as CSR
	raw_graph_to_AdjacencyList();
}

int main(int argc, char **argv)
{
	//显示处理器核心数
	int numProcs = omp_get_num_procs();
	//std::cout << "Number of available CPU cores: " << numProcs << std::endl;
	//num_threads = numProcs / 2;
	num_threads = 32;

	/*输入七个参数，不计算读文件（reading files）、分配内存（malloc）和初始化内存（memset）的时间。*/
	F0 = atoi(argv[1]);                    //输入层特征长度，其中atoi()函数用于将字符串转换为整数类型。
	F1 = atoi(argv[2]);                    //第一层特征长度
	F2 = atoi(argv[3]);                    //第二层特征长度
	readGraph(argv[4]);                    //图结构（文件名）  
	readFloat(argv[5], X0, v_num * F0);    //X0（输入顶点特征矩阵文件名），矩阵大小为“顶点数×F0”
	readFloat(argv[6], W1, F0 * F1);       //W1（第一层权重矩阵文件名），矩阵大小为“F0×F1”
	readFloat(argv[7], W2, F1 * F2);       //W2（第二层权重矩阵文件名），矩阵大小为“F1×F2”

	initFloat(X1, v_num * F1);
	initFloat(X1_inter, v_num * F1);
	initFloat(X2, v_num * F2);
	initFloat(X2_inter, v_num * F2);

	//计算开始时的时间点
	TimePoint start = chrono::steady_clock::now();

	//预处理的时间应包括在内
	somePreprocessing();
	edgeNormalization();

	// printf("Layer1 XW\n");
	XW(F0, F1, X0, X1_inter, W1);

	// printf("Layer1 AX\n");
	AX(F1, X1_inter, X1);

	// printf("Layer1 ReLU\n");
	ReLU(F1, X1);

	// printf("Layer2 XW\n");
	XW(F1, F2, X1, X2_inter, W2);

	// printf("Layer2 AX\n");
	AX(F2, X2_inter, X2);

	// printf("Layer2 LogSoftmax\n");
	LogSoftmax(F2, X2);

	// You need to compute the max row sum for result verification
	float max_sum = MaxRowSum(X2, F2);

	//计算结束时的时间点
	TimePoint end = chrono::steady_clock::now();
	chrono::duration<double> l_durationSec = end - start;
	double l_timeMs = l_durationSec.count() * 1e3;

	// Finally, the max row sum and the computing time
	// should be print to the terminal in the following format
	printf("%.8f\n", max_sum);
	printf("%.8lf\n", l_timeMs);
	fflush(stdout);

	// Remember to free your allocated memory
	freeFloats();
}