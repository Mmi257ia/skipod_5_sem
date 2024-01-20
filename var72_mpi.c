#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <time.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2*2*2+2)
float maxeps = 0.1e-7;
int itmax = 100;
int i, j, k, i_start, i_end, slice;
int processes, pid;
float eps;
float * A, * B, * A_outer, * B_outer;
struct timespec start, end;
MPI_Request req_send[2], req_recv[2];
MPI_Status stat[2];

void relax();
void resid();
void init();
void verify();

int main(int an, char **as)
{
	int it;
	MPI_Init(&an, &as);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	init();
	clock_gettime(CLOCK_REALTIME, &start);
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		// if (pid == 0)
		// 	printf("it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps) break;
	}
	verify();
	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_REALTIME, &end);
	double timespan = end.tv_sec + (double)end.tv_nsec / 1000000000 - start.tv_sec - (double)start.tv_nsec / 1000000000; 
	if (pid == 0)
		printf("Elapsed time: %lf\nUsed processes: %d\n", timespan, processes);
	free(A_outer);
	free(B_outer);
	MPI_Finalize();
	return 0;
}

void init()
{
	i_start = round((float)N / processes * pid); // start of process' slice
	i_end = round((float)N / processes * (pid + 1)); // start of the next process' slice
	slice = i_end - i_start;
	A_outer = malloc(sizeof(float) * N*N*(slice+4)); // allows to address below and above A and B for received layers
	B_outer = malloc(sizeof(float) * N*N*(slice+4));
	A = A_outer + N*N*2;
	B = B_outer + N*N*2;
	for(i = 0; i < slice; i++)
	for(j = 0; j < N; j++)
	for(k = 0; k < N; k++)
	{
		if((i + i_start) == 0  || (i + i_start) == N-1 || j == 0  || j == N-1 || k == 0 || k == N-1) A[N*N*i + N*j + k] = 0.;
		else A[N*N*i + N*j + k] = ( 4. + (i + i_start) + j + k);
	}
} 

void relax() // Assuming that N >= 2*processes
{
	int local_i_start = pid == 0 ? 2 : 0, local_i_end = pid == processes - 1 ? slice-2 : slice;

	if (processes > 1) {
		// sending our border layers to neighbors
		if (pid != 0)
			MPI_Isend(A, N*N*2, MPI_FLOAT, pid-1, 0, MPI_COMM_WORLD, &req_send[0]);
		if (pid != processes - 1)
			MPI_Isend(A + N*N*(slice-2), N*N*2, MPI_FLOAT, pid+1, 0, MPI_COMM_WORLD, &req_send[1]);

		// and receiving their from them
		if (pid != 0)
			MPI_Irecv(A - N*N*2, N*N*2, MPI_FLOAT, pid-1, 0, MPI_COMM_WORLD, &req_recv[0]);
		if (pid != processes - 1)
			MPI_Irecv(A + N*N*slice, N*N*2, MPI_FLOAT, pid+1, 0, MPI_COMM_WORLD, &req_recv[1]);

		// we need received values for relaxing
		if (pid == 0)
			MPI_Wait(&req_recv[1], &stat[0]);
		else if (pid == processes - 1)
			MPI_Wait(&req_recv[0], &stat[1]);
		else
			MPI_Waitall(2, req_recv, stat);
	}

	for(i = local_i_start; i < local_i_end; i++)
	for(j = 2; j < N-2; j++)
	for(k = 2; k < N-2; k++)
	{
		B[N*N*i + N*j + k]=(A[N*N*(i-1) + N*j + k] + A[N*N*(i+1) + N*j + k] + A[N*N*i + N*(j-1) + k] + A[N*N*i + N*(j+1) + k] +
							A[N*N*i + N*j + k-1]  +  A[N*N*i + N*j + k+1]  +  A[N*N*(i-2) + N*j + k] + A[N*N*(i+2) + N*j + k] +
							A[N*N*i + N*(j-2) + k] + A[N*N*i + N*(j+2) + k] + A[N*N*i + N*j + k-2]  +  A[N*N*i + N*j + k+2]) / 12.;
		// B[i][j][k]=(A[i-1][j][k] + A[i+1][j][k] + A[i][j-1][k] + A[i][j+1][k] + A[i][j][k-1] + A[i][j][k+1]+
		// 	A[i-2][j][k] + A[i+2][j][k] + A[i][j-2][k] + A[i][j+2][k] + A[i][j][k-2] + A[i][j][k+2])/12.;
	}

	// we will change A matrix so we have to finish sending by now
	if (processes > 1)
	if (pid == 0) {
		MPI_Wait(&req_send[1], &stat[0]);
	} else if (pid == processes - 1) {
		MPI_Wait(&req_send[0], &stat[1]);
	} else {
		MPI_Waitall(2, req_send, stat);
	}
}

void resid()
{
	int local_i_start = pid == 0 ? 1 : 0, local_i_end = pid == processes - 1 ? slice-1 : slice;
	for(i = local_i_start; i < local_i_end; i++)
	for(j = 1; j < N-1; j++)
	for(k = 1; k < N-1; k++)
	{
		float e;
		e = fabs(A[N*N*i + N*j + k] - B[N*N*i + N*j + k]);
		A[N*N*i + N*j + k] = B[N*N*i + N*j + k]; 
		eps = Max(eps,e);
	}

	float global_eps;
	MPI_Allreduce(&eps, &global_eps, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
	eps = global_eps;
	MPI_Barrier(MPI_COMM_WORLD);
}

void verify()
{
	double s = 0.;
	for(i = 0; i < slice; i++)
	for(j = 0; j < N; j++)
	for(k = 0; k < N; k++)
	{
		s += (double)A[N*N*i + N*j + k]*(i + i_start + 1)*(j + 1)*(k + 1)/(N*N*N);
	}
	double global_s;
	MPI_Allreduce(&s, &global_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (pid == 0)
		printf("  S = %lf\n", global_s);
}
