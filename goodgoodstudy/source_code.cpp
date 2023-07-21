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
#include <immintrin.h> // ����AVXָ���ͷ�ļ�
#pragma GCC target("avx2")
#pragma GCC optimize("O3","unroll-loops")

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;                   //������
int e_num = 0;                   //����
int F0 = 0, F1 = 0, F2 = 0;      //F0 <= 128; F1 = 16; F2 <= 32
int num_threads = 0;             //����ִ�е��߳���

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

//��ȡ�ļ�����ʹ��ʾ���ļ��ṩ��readGraph()�����������޸ģ�������ִ��ʱ���ڣ�����ת��Ϊ�ڽӱ����CSR�ȸ�ʽ����somePreprocessing()������ʵ�֣�������ִ��ʱ�䡣
void readGraph(char *fname)
{
	ifstream infile(fname);

	int source;     //Դ�ڵ�
	int end;        //Ŀ�Ľڵ�

	infile >> v_num >> e_num;     //�ļ���һ���Ƕ������ͱ���

	raw_graph.resize(e_num * 2);  //�ı�ԭͼ������Ϊ�����Ķ���

	while (!infile.eof())
	{
		infile >> source >> end;       //���ļ��еĵڶ��е����һ�У�ÿ������������������Դ�ڵ�id��Ŀ�Ľڵ�id
		if (infile.peek() == EOF)
			break;
		raw_graph.push_back(source);   //��Դ�ڵ�id��ӵ�ԭͼ������ĩβ
		raw_graph.push_back(end);      //��Ŀ�Ľڵ�id��ӵ�ԭͼ������ĩβ
	}
}

//��ԭͼ����תΪ�ڽӾ���
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

//�ߵĹ�һ��
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
		// �ļ���ʧ�ܣ����д�����
		printf("�޷����ļ���%s\n", fname);
		// �������������
		// ...
	}
	else
	{
		// �ļ��ɹ��򿪣����Խ����ļ���ȡ����
		// ��ȡ�ļ�����
		fread(dst, sizeof(float), num, fp);
		fclose(fp);
	}
}

void initFloat(float *&dst, int num)
{
	dst = (float *)malloc(num * sizeof(float));
	memset(dst, 0, num * sizeof(float));
}

/*���߳�
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

/*AVXָ�
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
	// ÿ�δ���8�������ȸ�����
	for (int i = 0; i < v_num; i++)
	{
		for (int j = 0; j < out_dim; j += 8)
		{
			__m256 sum = _mm256_setzero_ps(); // ��ʼ���ۼӺ�Ϊ������

			for (int k = 0; k < in_dim; k++)
			{
				// ������������Ȩ�ؾ��������
				__m256 X_vec = _mm256_loadu_ps(&in_X[i * in_dim + k]);
				__m256 W_vec = _mm256_loadu_ps(&W[k * out_dim + j]);

				// ִ�г˷�����
				__m256 mul = _mm256_mul_ps(X_vec, W_vec);

				// �ۼӳ˻����
				sum = _mm256_add_ps(sum, mul);
			}

			// �洢�ۼӺ͵��������
			_mm256_storeu_ps(&out_X[i * out_dim + j], sum);
		}
	}
}*/

/*���߳�+AVXָ�*/
void XW(int in_dim, int out_dim, float* in_X, float* out_X, float* W)
{
	omp_set_num_threads(num_threads);

#pragma omp parallel for
	for (int i = 0; i < v_num; i++)
	{
		for (int j = 0; j < out_dim; j += 8)
		{
			__m256 sum = _mm256_setzero_ps(); // ��ʼ���ۼӺ�Ϊ������

			for (int k = 0; k < in_dim; k++)
			{
				__m256 X_vec = _mm256_loadu_ps(&in_X[i * in_dim + k]);
				__m256 W_vec = _mm256_loadu_ps(&W[k * out_dim + j]);

				__m256 mul = _mm256_mul_ps(X_vec, W_vec);

				sum = _mm256_add_ps(sum, mul);
			}

			// �洢�ۼӺ͵��������
			_mm256_storeu_ps(&out_X[i * out_dim + j], sum);
		}
	}
}


/*���߳�
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

/*AVXָ�
void AX(int dim, float *in_X, float *out_X)
{
	for (int i = 0; i < v_num; i++)
	{
		std::vector<int> &nlist = edge_index[i];
		for (int j = 0; j < nlist.size(); j += 8) // ����ʹ��AVX-256��һ�δ���8��Ԫ��
		{
			int nbr[8];
			for (int k = 0; k < 8; k++)
			{
				if (j + k < nlist.size())
					nbr[k] = nlist[j + k];
				else
					nbr[k] = -1; // ���ڴ���߽����������ھ���������8���������-1
			}

			for (int k = 0; k < dim; k += 8) // һ�δ���8��ά��
			{
				__m256 in_vec = _mm256_loadu_ps(&in_X[i * dim + k]); // �������������е�����
				__m256 out_vec = _mm256_loadu_ps(&out_X[i * dim + k]); // ������������е�����

				for (int l = 0; l < 8; l++)
				{
					if (nbr[l] != -1)
					{
						__m256 edge_val_vec = _mm256_set1_ps(edge_val[i][j + l]); // ʹ�ñ߽�����е�ֵ����һ������
						__m256 in_X_nbr_vec = _mm256_loadu_ps(&in_X[nbr[l] * dim + k]); // �����ھӽڵ��������������

						__m256 result_vec = _mm256_mul_ps(in_X_nbr_vec, edge_val_vec); // ��ӦԪ�����
						out_vec = _mm256_add_ps(out_vec, result_vec); // �ۼӵ����������
					}
				}

				_mm256_storeu_ps(&out_X[i * dim + k], out_vec); // �洢���º�������������
			}
		}
	}
}*/

/*���߳�+AVXָ�*/
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
	//��ʾ������������
	int numProcs = omp_get_num_procs();
	//std::cout << "Number of available CPU cores: " << numProcs << std::endl;
	//num_threads = numProcs / 2;
	num_threads = 32;

	/*�����߸���������������ļ���reading files���������ڴ棨malloc���ͳ�ʼ���ڴ棨memset����ʱ�䡣*/
	F0 = atoi(argv[1]);                    //������������ȣ�����atoi()�������ڽ��ַ���ת��Ϊ�������͡�
	F1 = atoi(argv[2]);                    //��һ����������
	F2 = atoi(argv[3]);                    //�ڶ�����������
	readGraph(argv[4]);                    //ͼ�ṹ���ļ�����  
	readFloat(argv[5], X0, v_num * F0);    //X0�����붥�����������ļ������������СΪ����������F0��
	readFloat(argv[6], W1, F0 * F1);       //W1����һ��Ȩ�ؾ����ļ������������СΪ��F0��F1��
	readFloat(argv[7], W2, F1 * F2);       //W2���ڶ���Ȩ�ؾ����ļ������������СΪ��F1��F2��

	initFloat(X1, v_num * F1);
	initFloat(X1_inter, v_num * F1);
	initFloat(X2, v_num * F2);
	initFloat(X2_inter, v_num * F2);

	//���㿪ʼʱ��ʱ���
	TimePoint start = chrono::steady_clock::now();

	//Ԥ�����ʱ��Ӧ��������
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

	//�������ʱ��ʱ���
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