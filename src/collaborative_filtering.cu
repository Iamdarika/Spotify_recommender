// src/collaborative_filtering.cu
// Compile with: nvcc -O2 -Xcompiler -fopenmp -o cf_cuda src/collaborative_filtering.cu
// This code implements Collaborative Filtering using CUDA to compute cosine similarity
// for a set of tracks. Each track has F=8 features (from preprocessing).

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Constants
#define MAX_LINE 2048      // Max characters per CSV line
#define F 8                // Number of features per track (z-score features from preprocessing)

// ------------------- CUDA Kernel -------------------
// Each thread computes cosine similarity between track i and track j
__global__ void cosineKernel(const float* __restrict__ X, int N, float* __restrict__ S){
    // Compute track indices based on thread & block indices
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // Row track index
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // Column track index

    // If index exceeds number of tracks, exit
    if(i>=N || j>=N) return;

    // Variables to store dot product and magnitudes
    float dot=0, ni=0, nj=0;

    // Loop over features (F=8)
    #pragma unroll
    for(int k=0;k<F;++k){
        float a = X[i*F+k];  // Feature k of track i
        float b = X[j*F+k];  // Feature k of track j
        dot += a*b;           // Compute dot product
        ni += a*a;            // Compute ||i||^2
        nj += b*b;            // Compute ||j||^2
    }

    // Compute cosine similarity: dot / (||i||*||j||)
    float sim = 0.f;
    if(ni>0 && nj>0) sim = dot / (sqrtf(ni)*sqrtf(nj));

    // Store similarity in similarity matrix S[i*N + j]
    S[i*N + j] = sim;
}

// ------------------- CSV Parsing -------------------
// Reads CSV and extracts features, track IDs, names
static int read_sample(const char* path, char ids[][64], char names[][128], float** outX, int* outN){
    FILE* fp = fopen(path, "r");  // Open CSV file
    if(!fp){ fprintf(stderr,"[cuda-cf] cannot open %s\n", path); return 0; }

    char line[MAX_LINE];
    if(!fgets(line,sizeof(line),fp)){ fclose(fp); return 0; } // Skip CSV header

    int cap=16, n=0;  // Initial capacity for tracks
    float* X = (float*)malloc(cap*F*sizeof(float)); // Allocate memory for features

    while(fgets(line,sizeof(line),fp)){ // Read CSV line by line
        if(n==cap){                  // If capacity exceeded, double it
            cap*=2; 
            X=(float*)realloc(X,cap*F*sizeof(float)); 
        }

        char *tok, *save=NULL; 
        int col=0; 
        float f[F];       // Temporary array for features of current track
        int fk=0;         // Feature counter
        char id[64]="", nm[128]="";  // Temporary ID and name

        // Tokenize CSV line by commas
        for(tok=strtok_r(line,",",&save); tok; tok=strtok_r(NULL,",",&save),++col){
            while(*tok==' '||*tok=='\t') tok++;  // Remove leading spaces
            size_t L=strlen(tok);
            while(L && (tok[L-1]=='\n'||tok[L-1]=='\r'||tok[L-1]==' '||tok[L-1]=='\t')) tok[--L]='\0'; // Remove trailing spaces

            if(col==0) strncpy(id,tok,sizeof(id)-1);       // Column 0 -> track ID
            else if(col==1) strncpy(nm,tok,sizeof(nm)-1);  // Column 1 -> track name
            else if(fk<F) f[fk++]=(float)atof(tok);       // Next F columns -> features
        }

        // Only store track if all F features are present
        if(fk==F){
            if(n<1024){ // Store ID & name if within limit
                strncpy(ids[n],id,63); ids[n][63]='\0';
                strncpy(names[n],nm,127); names[n][127]='\0';
            }
            for(int k=0;k<F;++k) X[n*F+k]=f[k];  // Copy features to X
            n++;
        }
    }

    fclose(fp);
    *outX = X; *outN = n;  // Return features & track count
    return n>0;
}

// ------------------- Main Collaborative Filtering Function -------------------
extern "C" void collaborative_filtering(){
    // CSV file path
    const char* path = "data/sample_spotify_normalized.csv";

    // Host arrays for IDs, names
    char ids[1024][64]; 
    char names[1024][128];

    float* hX=NULL;  // Features on CPU
    int N=0;         // Number of tracks

    // Read CSV
    if(!read_sample(path, ids, names, &hX, &N)){
        printf("[cuda-cf] normalized file not found; run preprocessing first.\n");
        return;
    }
    printf("[cuda-cf] Loaded %d tracks from %s\n", N, path);

    // ------------------- GPU Memory Allocation -------------------
    float *dX=NULL, *dS=NULL;
    cudaMalloc(&dX, N*F*sizeof(float));   // Allocate GPU memory for features
    cudaMalloc(&dS, N*N*sizeof(float));   // Allocate GPU memory for similarity matrix
    cudaMemcpy(dX, hX, N*F*sizeof(float), cudaMemcpyHostToDevice); // Copy features to GPU

    // ------------------- Launch CUDA Kernel -------------------
    dim3 block(16,16);                     // 16x16 threads per block
    dim3 grid((N+15)/16, (N+15)/16);       // Number of blocks
    cosineKernel<<<grid,block>>>(dX, N, dS);  // Launch kernel
    cudaDeviceSynchronize();               // Wait for GPU to finish

    // ------------------- Copy Similarity Matrix back to CPU -------------------
    float* hS=(float*)malloc(N*N*sizeof(float));  
    cudaMemcpy(hS, dS, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // ------------------- Find Top-1 Similar Track -------------------
    for(int i=0;i<N;++i){
        int best=-1; float bestv=-2.f;
        for(int j=0;j<N;++j){
            if(i==j) continue;               // Skip same track
            float v=hS[i*N+j];               // Get similarity
            if(v>bestv){ bestv=v; best=j; } // Track with max similarity
        }
        if(best>=0){
            printf("[cuda-cf] %-12s -> Top1 similar: %-12s (sim=%.3f)\n",
                   ids[i], ids[best], bestv);  // Print result
        }
    }

    // ------------------- Free Memory -------------------
    free(hS); cudaFree(dS); cudaFree(dX); free(hX);
    printf("[cuda-cf] Done.\n");
}
