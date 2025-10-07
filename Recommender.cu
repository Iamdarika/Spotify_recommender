#include "Recommender.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <queue>
#include <cctype>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - Status: " << status << std::endl; \
            if (status == CUBLAS_STATUS_NOT_INITIALIZED) std::cerr << "  CUBLAS_STATUS_NOT_INITIALIZED" << std::endl; \
            if (status == CUBLAS_STATUS_ALLOC_FAILED) std::cerr << "  CUBLAS_STATUS_ALLOC_FAILED" << std::endl; \
            if (status == CUBLAS_STATUS_INVALID_VALUE) std::cerr << "  CUBLAS_STATUS_INVALID_VALUE" << std::endl; \
            if (status == CUBLAS_STATUS_ARCH_MISMATCH) std::cerr << "  CUBLAS_STATUS_ARCH_MISMATCH" << std::endl; \
            return false; \
        } \
    } while(0)

// CUDA kernel to compute norms of feature vectors
__global__ void computeNormsKernel(const float* features, float* norms, int numSongs, int featureCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numSongs) {
        float sum = 0.0f;
        for (int i = 0; i < featureCount; ++i) {
            float val = features[idx * featureCount + i];
            sum += val * val;
        }
        norms[idx] = sqrtf(sum);
    }
}

// CUDA kernel to normalize similarity scores by norms (final step of cosine similarity)
__global__ void normalizeSimilaritiesKernel(float* similarities, const float* norms, 
                                            float queryNorm, int numSongs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numSongs) {
        float denominator = norms[idx] * queryNorm;
        if (denominator > 1e-8f) {
            similarities[idx] = similarities[idx] / denominator;
        } else {
            similarities[idx] = 0.0f;
        }
        
        // Clamp to [-1, 1] to handle floating point errors
        similarities[idx] = fminf(1.0f, fmaxf(-1.0f, similarities[idx]));
    }
}

Recommender::Recommender() 
    : initialized(false), numSongs(0), d_features(nullptr), 
      d_queryFeature(nullptr), d_similarities(nullptr), cublasHandle(nullptr) {
}

Recommender::~Recommender() {
    if (d_features) cudaFree(d_features);
    if (d_queryFeature) cudaFree(d_queryFeature);
    if (d_similarities) cudaFree(d_similarities);
    if (cublasHandle) {
        cublasDestroy(static_cast<cublasHandle_t>(cublasHandle));
    }
}

bool Recommender::initialize(const std::vector<Song>& songs) {
    std::cout << "Initializing GPU-accelerated recommender..." << std::endl;
    
    if (songs.empty()) {
        std::cerr << "Error: Empty song database" << std::endl;
        return false;
    }
    
    songDatabase = songs;
    numSongs = songs.size();
    
    // Check for CUDA devices
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: No CUDA devices found or CUDA not available" << std::endl;
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublasHandle = static_cast<void*>(handle);
    
    // Allocate GPU memory
    size_t featureMatrixSize = numSongs * FEATURE_COUNT * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_features, featureMatrixSize));
    CUDA_CHECK(cudaMalloc(&d_queryFeature, FEATURE_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_similarities, numSongs * sizeof(float)));
    
    // Prepare feature matrix in row-major format
    std::vector<float> featureMatrix(numSongs * FEATURE_COUNT);
    for (int i = 0; i < numSongs; ++i) {
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            featureMatrix[i * FEATURE_COUNT + j] = songs[i].features[j];
        }
    }
    
    // Copy feature matrix to GPU
    CUDA_CHECK(cudaMemcpy(d_features, featureMatrix.data(), 
                         featureMatrixSize, cudaMemcpyHostToDevice));
    
    std::cout << "Successfully initialized with " << numSongs << " songs on GPU" << std::endl;
    
    initialized = true;
    return true;
}

void Recommender::calculateSimilarities(int queryIndex, float* similarities) {
    if (!initialized || queryIndex < 0 || queryIndex >= numSongs) {
        std::cerr << "Error: Invalid query index or recommender not initialized" << std::endl;
        return;
    }
    
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublasHandle);
    
    // Copy query feature to device
    float* queryFeatureHost = songDatabase[queryIndex].features;
    cudaMemcpy(d_queryFeature, queryFeatureHost, 
               FEATURE_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute dot products using cuBLAS SGEMV
    // similarities = features * queryFeature
    // features is (numSongs x FEATURE_COUNT) matrix
    // queryFeature is (FEATURE_COUNT x 1) vector
    // result is (numSongs x 1) vector
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // cuBLAS uses column-major, but we can treat our row-major as transposed
    // We want: d_similarities = d_features * d_queryFeature
    // In column-major interpretation: d_similarities = d_features^T * d_queryFeature
    // So we use SGEMV with transpose
    cublasSgemv(handle, CUBLAS_OP_T,
                FEATURE_COUNT, numSongs,  // Transposed dimensions
                &alpha,
                d_features, FEATURE_COUNT,  // Leading dimension in row-major
                d_queryFeature, 1,
                &beta,
                d_similarities, 1);
    
    // Allocate device memory for norms
    float* d_norms;
    cudaMalloc(&d_norms, numSongs * sizeof(float));
    
    // Compute norms of all feature vectors
    int blockSize = 256;
    int numBlocks = (numSongs + blockSize - 1) / blockSize;
    computeNormsKernel<<<numBlocks, blockSize>>>(d_features, d_norms, numSongs, FEATURE_COUNT);
    
    // Compute norm of query vector
    float queryNorm = 0.0f;
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        queryNorm += queryFeatureHost[i] * queryFeatureHost[i];
    }
    queryNorm = std::sqrt(queryNorm);
    
    // Normalize similarities to get cosine similarity
    normalizeSimilaritiesKernel<<<numBlocks, blockSize>>>(d_similarities, d_norms, 
                                                          queryNorm, numSongs);
    
    // Copy results back to host
    cudaMemcpy(similarities, d_similarities, numSongs * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Free temporary memory
    cudaFree(d_norms);
}

std::vector<int> Recommender::recommendByIndex(int songIndex, int topN) {
    if (!initialized) {
        std::cerr << "Error: Recommender not initialized" << std::endl;
        return {};
    }
    
    if (songIndex < 0 || songIndex >= numSongs) {
        std::cerr << "Error: Invalid song index: " << songIndex << std::endl;
        return {};
    }
    
    // Allocate host memory for similarities
    std::vector<float> similarities(numSongs);
    
    // Calculate similarities on GPU
    calculateSimilarities(songIndex, similarities.data());
    
    // Use a min-heap to find top-N recommendations
    std::priority_queue<Recommendation> heap;
    
    for (int i = 0; i < numSongs; ++i) {
        if (i == songIndex) continue; // Skip the query song itself
        
        Recommendation rec(i, similarities[i]);
        
        if (heap.size() < static_cast<size_t>(topN)) {
            heap.push(rec);
        } else if (rec.similarity > heap.top().similarity) {
            heap.pop();
            heap.push(rec);
        }
    }
    
    // Extract results and reverse to get descending order
    std::vector<int> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top().songIndex);
        heap.pop();
    }
    std::reverse(results.begin(), results.end());
    
    return results;
}

int Recommender::findSongByTrackId(const std::string& trackId) const {
    for (int i = 0; i < numSongs; ++i) {
        if (songDatabase[i].track_id == trackId) {
            return i;
        }
    }
    return -1;
}

std::string Recommender::toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

int Recommender::findSongByName(const std::string& trackName) const {
    std::string lowerQuery = toLower(trackName);
    
    // First try exact match (case-insensitive)
    for (int i = 0; i < numSongs; ++i) {
        if (toLower(songDatabase[i].track_name) == lowerQuery) {
            return i;
        }
    }
    
    // If no exact match, try substring match
    for (int i = 0; i < numSongs; ++i) {
        if (toLower(songDatabase[i].track_name).find(lowerQuery) != std::string::npos) {
            return i;
        }
    }
    
    return -1;
}

std::vector<int> Recommender::recommend(const std::string& trackId, int topN) {
    int index = findSongByTrackId(trackId);
    if (index == -1) {
        std::cerr << "Error: Song with track_id '" << trackId << "' not found" << std::endl;
        return {};
    }
    return recommendByIndex(index, topN);
}

std::vector<int> Recommender::recommendByName(const std::string& trackName, int topN) {
    int index = findSongByName(trackName);
    if (index == -1) {
        std::cerr << "Error: Song with name '" << trackName << "' not found" << std::endl;
        return {};
    }
    return recommendByIndex(index, topN);
}

