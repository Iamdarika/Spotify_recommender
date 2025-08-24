// src/main.c (snippet)
#include <stdio.h>
#include <omp.h>

void preprocess_data();
void matrix_factorization();      // still stub ok for now
void collaborative_filtering();   // CUDA .cu provides this
void benchmark_parallel();        // MPI

int main(){
  printf("Starting Spotify HPC Recommender...\n");
  #pragma omp parallel
  { printf("OMP thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads()); }

  const char* p = getenv("SPOTIFY_CSV");
  printf("Input CSV: %s (default=data/sample_spotify.csv)\n", (p&&*p)?p:"data/sample_spotify.csv");

  preprocess_data();          // OpenMP
  benchmark_parallel();       // MPI
  collaborative_filtering();  // CUDA
  matrix_factorization();     // (ok as placeholder)

  printf("Done.\n");
  return 0;
}
