// src/collaborative_filtering.cu
// nvcc -O2 -Xcompiler -fopenmp -o cf_cuda src/collaborative_filtering.cu
// (this file includes host CSV parsing and a CUDA kernel for cosine sim)
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 2048
#define F 8  // number of z-score features we wrote from preprocessing

__global__ void cosineKernel(const float* __restrict__ X, int N, float* __restrict__ S){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N || j>=N) return;

    float dot=0, ni=0, nj=0;
    #pragma unroll
    for(int k=0;k<F;++k){
        float a = X[i*F+k];
        float b = X[j*F+k];
        dot += a*b;
        ni += a*a;
        nj += b*b;
    }
    float sim = 0.f;
    if(ni>0 && nj>0) sim = dot / (sqrtf(ni)*sqrtf(nj));
    S[i*N + j] = sim;
}

static int read_sample(const char* path, char ids[][64], char names[][128], float** outX, int* outN){
    FILE* fp = fopen(path, "r");
    if(!fp){ fprintf(stderr,"[cuda-cf] cannot open %s\n", path); return 0; }
    char line[MAX_LINE];
    if(!fgets(line,sizeof(line),fp)){ fclose(fp); return 0; } // header

    int cap=16, n=0;
    float* X = (float*)malloc(cap*F*sizeof(float));
    while(fgets(line,sizeof(line),fp)){
        if(n==cap){ cap*=2; X=(float*)realloc(X,cap*F*sizeof(float)); }
        char *tok, *save=NULL; int col=0; float f[F]; int fk=0;
        char id[64]="", nm[128]="";
        for(tok=strtok_r(line,",",&save); tok; tok=strtok_r(NULL,",",&save),++col){
            while(*tok==' '||*tok=='\t') tok++;
            size_t L=strlen(tok);
            while(L && (tok[L-1]=='\n'||tok[L-1]=='\r'||tok[L-1]==' '||tok[L-1]=='\t')) tok[--L]='\0';

            if(col==0) strncpy(id,tok,sizeof(id)-1);
            else if(col==1) strncpy(nm,tok,sizeof(nm)-1);
            else if(fk<F) f[fk++]=(float)atof(tok);
        }
        if(fk==F){
            if(n<1024){ strncpy(ids[n],id,63); ids[n][63]='\0'; strncpy(names[n],nm,127); names[n][127]='\0'; }
            for(int k=0;k<F;++k) X[n*F+k]=f[k];
            n++;
        }
    }
    fclose(fp);
    *outX = X; *outN = n;
    return n>0;
}

extern "C" void collaborative_filtering(){
    const char* path = "data/sample_spotify_normalized.csv";
    char ids[1024][64]; char names[1024][128];
    float* hX=NULL; int N=0;
    if(!read_sample(path, ids, names, &hX, &N)){
        printf("[cuda-cf] normalized file not found; run preprocessing first.\n");
        return;
    }
    printf("[cuda-cf] Loaded %d tracks from %s\n", N, path);

    float *dX=NULL, *dS=NULL;
    cudaMalloc(&dX, N*F*sizeof(float));
    cudaMalloc(&dS, N*N*sizeof(float));
    cudaMemcpy(dX, hX, N*F*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16,16), grid((N+15)/16, (N+15)/16);
    cosineKernel<<<grid,block>>>(dX, N, dS);
    cudaDeviceSynchronize();

    float* hS=(float*)malloc(N*N*sizeof(float));
    cudaMemcpy(hS, dS, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<N;++i){
        int best=-1; float bestv=-2.f;
        for(int j=0;j<N;++j){
            if(i==j) continue;
            float v=hS[i*N+j];
            if(v>bestv){ bestv=v; best=j; }
        }
        if(best>=0){
            printf("[cuda-cf] %-12s -> Top1 similar: %-12s (sim=%.3f)\n",
                   ids[i], ids[best], bestv);
        }
    }

    free(hS); cudaFree(dS); cudaFree(dX); free(hX);
    printf("[cuda-cf] Done.\n");
}
