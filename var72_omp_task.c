#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2*2*2*2+2)
float   maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
float eps;
float * eps_threads;
float A [N][N][N],  B [N][N][N];

void relax();
void resid();
void init();
void verify(); 

int main(int an, char **as)
{
	int it;
	double start, end;
	init();
	start = omp_get_wtime();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		//printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	verify();
	end = omp_get_wtime();
	printf("Elapsed time: %f\nUsed threads: %d\n", end - start, omp_get_max_threads());
	return 0;
}

void init() // Changed iteration order from k-j-i to i-j-k
{ 
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
	eps_threads = calloc(omp_get_max_threads(), sizeof(float));
} 

void relax() // Changed iteration order from k-j-i to i-j-k
{
	#pragma omp parallel default(none) shared(A, B) private(i, j)
	#pragma omp master
		for(i=2; i<=N-3; i++)
		for(j=2; j<=N-3; j++)
		#pragma omp task default(none) private(k) firstprivate(i, j) shared(A, B)
		for(k=2; k<=N-3; k++)
		{
			B[i][j][k] = (A[i-1][j][k] + A[i+1][j][k] + A[i][j-1][k] + A[i][j+1][k] + A[i][j][k-1] + A[i][j][k+1]+
				A[i-2][j][k] + A[i+2][j][k] + A[i][j-2][k] + A[i][j+2][k] + A[i][j][k-2] + A[i][j][k+2])/12.;
		}
	#pragma omp taskwait
}

void resid() // Changed iteration order from k-j-i to i-j-k
{
	#pragma omp parallel default(none) shared(A, B, eps_threads) private(i, j)
	#pragma omp master
		for(i=1; i<=N-2; i++)
		for(j=1; j<=N-2; j++)
		#pragma omp task default(none) private(k) firstprivate(i, j) shared(A, B, eps_threads)
		{
			float e_task[N-2];
			int threadnum = omp_get_thread_num();
			for(k=1; k<=N-2; k++)
			{
				e_task[k-1] = fabs(A[i][j][k] - B[i][j][k]);
				A[i][j][k] = B[i][j][k];
			}
			for(k=1; k<=N-2; k++)
				eps_threads[threadnum] = Max(eps_threads[threadnum], e_task[k-1]);
		}
	#pragma omp taskwait
	for (i=0; i<omp_get_max_threads(); i++)
		eps = Max(eps, eps_threads[i]);
	memset(eps_threads, 0, omp_get_max_threads() * sizeof(float));
}

void verify()
{
	float s;
	s=0.;
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	printf("  S = %f\n",s);
}
