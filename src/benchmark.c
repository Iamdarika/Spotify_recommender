#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

static double wtime(){ return MPI_Wtime(); }

void benchmark_parallel() {
    int rank,size;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Problem size (can be tuned)
    const long long N = 50LL * 1000LL * 1000LL; // 50M elements (conceptual demo)
    long long chunk = N / size;
    long long start = rank * chunk;
    long long end   = (rank==size-1) ? N : start + chunk;

    // Each rank computes its partial dot-product in parallel with OpenMP
    double t0 = wtime();
    double local = 0.0;
    #pragma omp parallel for reduction(+:local) schedule(static)
    for(long long i=start;i<end;++i){
        // simple deterministic vectors: a[i]=1/(i+1), b[i]=1
        local += 1.0 / (double)(i+1);
    }
    double t1 = wtime();

    double global=0.0;
    MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    if(rank==0){
        printf("[mpi] processes=%d  N=%lld\n", size, N);
        printf("[mpi] global dot=%.6f  time=%.3fs\n", global, t1 - t0);
        printf("[mpi] note: OpenMP used inside each rank for CPU parallelism\n");
    }
    MPI_Finalize();
}


