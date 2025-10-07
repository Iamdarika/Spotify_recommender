# System Workflow Diagrams

## 1. High-Level System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INPUT                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
            ┌───────▼─────┐   ┌──────▼──────┐
            │ PREPROCESS  │   │ RECOMMEND   │
            │    MODE     │   │    MODE     │
            └───────┬─────┘   └──────┬──────┘
                    │                │
                    │                │
    ┌───────────────▼───────────┐    │
    │   DataManager              │    │
    │   - Parse CSV             │    │
    │   - Clean Data            │    │
    │   - Normalize Features    │    │
    │   - Encode Genres         │    │
    │   - Parallelize (OpenMP)  │    │
    └───────────┬───────────────┘    │
                │                    │
                ▼                    │
    ┌─────────────────────┐          │
    │  songs_data.bin     │◄─────────┤
    │  (Binary Format)    │          │
    └─────────────────────┘          │
                                     │
                    ┌────────────────▼──────────────┐
                    │   Recommender                 │
                    │   - Load Binary Data          │
                    │   - Transfer to GPU           │
                    │   - Calculate Similarities    │
                    │   - Use cuBLAS (SGEMV)       │
                    │   - Custom CUDA Kernels       │
                    │   - Return Top-N             │
                    └────────────┬──────────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │   Console Output          │
                    │   - Query Song Info       │
                    │   - Top N Recommendations │
                    │   - Artist & Genre Info   │
                    └───────────────────────────┘
```

---

## 2. Preprocessing Pipeline

```
CSV File
  │
  ├─► Read Header
  │   └─► Map Column Names to Indices
  │
  ├─► Read All Lines
  │   └─► Store in Memory
  │
  ├─► Parallel Processing (OpenMP)
  │   │
  │   ├─► Thread 1: Parse rows 0-999
  │   ├─► Thread 2: Parse rows 1000-1999
  │   ├─► Thread 3: Parse rows 2000-2999
  │   └─► Thread N: Parse rows ...
  │       │
  │       ├─► Extract track_id, track_name, artists
  │       ├─► Extract 11 numerical features
  │       ├─► Extract genre (text)
  │       └─► Validate data
  │
  ├─► Merge Results
  │   └─► Create Genre ID Map
  │
  ├─► Calculate Min/Max for Each Feature
  │
  ├─► Parallel Normalization (OpenMP)
  │   │
  │   ├─► Thread 1: Normalize rows 0-999
  │   ├─► Thread 2: Normalize rows 1000-1999
  │   └─► Thread N: Normalize rows ...
  │       │
  │       └─► feature_norm = (value - min) / (max - min)
  │
  ├─► Binary Serialization
  │   │
  │   ├─► Write: numSongs, numGenres
  │   ├─► Write: Genre Mapping
  │   └─► Write: All Songs (metadata + features)
  │
  └─► songs_data.bin ✓
```

---

## 3. GPU Recommendation Pipeline

```
Query: "Song Name"
  │
  ├─► Load Binary Data (if not loaded)
  │   │
  │   ├─► Read Genre Mapping
  │   └─► Read All Songs
  │
  ├─► Initialize GPU (first time only)
  │   │
  │   ├─► Create cuBLAS Handle
  │   ├─► Allocate d_features (N × 12)
  │   ├─► Allocate d_queryFeature (1 × 12)
  │   ├─► Allocate d_similarities (N × 1)
  │   └─► Copy Feature Matrix to GPU
  │
  ├─► Find Query Song Index
  │   │
  │   ├─► Search by track_id or track_name
  │   └─► Return index or -1 (not found)
  │
  ├─► Calculate Similarities (GPU)
  │   │
  │   ├─► Copy Query Features to GPU
  │   │   └─► cudaMemcpy(d_queryFeature)
  │   │
  │   ├─► Compute Dot Products (cuBLAS)
  │   │   │
  │   │   └─► SGEMV: d_similarities = d_features^T × d_queryFeature
  │   │       │
  │   │       │   GPU Computation:
  │   │       │   ┌────────────────────────────────────┐
  │   │       │   │  Warp 1  ┌─────────────────────┐  │
  │   │       │   │  Warp 2  │ Parallel Multiply   │  │
  │   │       │   │  Warp 3  │ & Accumulate        │  │
  │   │       │   │  ...     │ for Each Song       │  │
  │   │       │   │  Warp N  └─────────────────────┘  │
  │   │       │   └────────────────────────────────────┘
  │   │       │
  │   │       └─► Result: Dot products for all songs
  │   │
  │   ├─► Compute Norms (Custom CUDA Kernel)
  │   │   │
  │   │   └─► computeNormsKernel<<<blocks, 256>>>
  │   │       │
  │   │       │   Each Thread:
  │   │       │   ┌────────────────────────────┐
  │   │       │   │ sum = 0                    │
  │   │       │   │ for i in 0..11:            │
  │   │       │   │     sum += f[i] * f[i]     │
  │   │       │   │ norm = sqrt(sum)           │
  │   │       │   └────────────────────────────┘
  │   │       │
  │   │       └─► Result: Norms for all songs
  │   │
  │   └─► Normalize Similarities (Custom CUDA Kernel)
  │       │
  │       └─► normalizeSimilaritiesKernel<<<blocks, 256>>>
  │           │
  │           │   Each Thread:
  │           │   ┌────────────────────────────────────┐
  │           │   │ cosine_sim = dot_product /         │
  │           │   │              (norm_A * norm_B)     │
  │           │   │ clamp(cosine_sim, -1, 1)          │
  │           │   └────────────────────────────────────┘
  │           │
  │           └─► Result: Cosine similarities [-1, 1]
  │
  ├─► Copy Results to CPU
  │   └─► cudaMemcpy(similarities, d_similarities)
  │
  ├─► Select Top-N (Min-Heap)
  │   │
  │   │   For each song (except query):
  │   │   ┌────────────────────────────────────┐
  │   │   │ if heap.size < N:                  │
  │   │   │     heap.push(song, similarity)    │
  │   │   │ else if similarity > heap.top():   │
  │   │   │     heap.pop()                     │
  │   │   │     heap.push(song, similarity)    │
  │   │   └────────────────────────────────────┘
  │   │
  │   └─► Result: Top N most similar songs
  │
  └─► Display Results
      │
      ├─► Show Query Song Info
      └─► Show Recommendations
          ├─► Song 1: Name, Artist, Genre
          ├─► Song 2: Name, Artist, Genre
          └─► Song N: Name, Artist, Genre
```

---

## 4. Memory Layout Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          HOST (CPU) MEMORY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  std::vector<Song> songDatabase                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Song 0:                                                │    │
│  │   track_id:    "7qiZfU4dY4fObP7Y8kDH3j"               │    │
│  │   track_name:  "Shape of You"                         │    │
│  │   artists:     "Ed Sheeran"                           │    │
│  │   genre_id:    42                                     │    │
│  │   features[12]: [0.825, 0.652, 0.583, ...]           │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │ Song 1:                                                │    │
│  │   track_id:    "3n3Ppam7vgaVa1iaRUc9Lp"               │    │
│  │   track_name:  "Blinding Lights"                      │    │
│  │   artists:     "The Weeknd"                           │    │
│  │   genre_id:    28                                     │    │
│  │   features[12]: [0.514, 0.730, 0.801, ...]           │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │ ...                                                    │    │
│  │ Song N-1                                               │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  std::map<int, std::string> genreMap                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 0  → "acoustic"                                        │    │
│  │ 1  → "afrobeat"                                        │    │
│  │ 2  → "alt-rock"                                        │    │
│  │ ...                                                    │    │
│  │ 113 → "world-music"                                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ cudaMemcpy (initialization)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DEVICE (GPU) MEMORY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  d_features: float[N × 12]  (N = number of songs)               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Row 0: [0.825, 0.652, 0.583, 0.732, 0.411, ...]       │    │
│  │ Row 1: [0.514, 0.730, 0.801, 0.623, 0.337, ...]       │    │
│  │ Row 2: [0.691, 0.572, 0.448, 0.889, 0.512, ...]       │    │
│  │ ...                                                    │    │
│  │ Row N-1: [...]                                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  d_queryFeature: float[12]  (Query song's features)             │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ [0.825, 0.652, 0.583, 0.732, 0.411, 0.298, ...]       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  d_similarities: float[N]  (Similarity scores)                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ [0.987, 0.956, 0.923, 0.891, 0.876, 0.845, ...]       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  d_norms: float[N]  (L2 norms, temporary)                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ [2.451, 2.389, 2.512, 2.378, 2.401, ...]              │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Thread Execution Model

### OpenMP (CPU Preprocessing)

```
Main Thread
    │
    ├─► Initialize Data Structures
    │
    ├─► #pragma omp parallel
    │   ┌────────────────────────────────────────────────────┐
    │   │                                                    │
    │   │  Thread 0  Thread 1  Thread 2  ...  Thread N-1    │
    │   │     │         │         │              │           │
    │   │     ├─► Rows  ├─► Rows  ├─► Rows      ├─► Rows    │
    │   │     │   0-999 │   1K-2K │   2K-3K      │   ...     │
    │   │     │         │         │              │           │
    │   │     ├─► Parse ├─► Parse ├─► Parse      ├─► Parse  │
    │   │     ├─► Clean ├─► Clean ├─► Clean      ├─► Clean  │
    │   │     └─► Store └─► Store └─► Store      └─► Store  │
    │   │                                                    │
    │   └────────────────────────────────────────────────────┘
    │
    ├─► #pragma omp barrier (Wait for all threads)
    │
    ├─► Merge Results
    │
    └─► Write to Binary File
```

### CUDA (GPU Similarity Calculation)

```
CPU (Host)                         GPU (Device)
    │
    ├─► Launch Kernel: computeNormsKernel<<<blocks, 256>>>
    │                          │
    │                          ├─► Grid (Multiple Blocks)
    │                          │   ┌──────────────────────────────┐
    │                          │   │ Block 0                      │
    │                          │   │  ┌─────────────────────┐    │
    │                          │   │  │ Warp 0 (32 threads) │    │
    │                          │   │  │ Warp 1 (32 threads) │    │
    │                          │   │  │ ...                 │    │
    │                          │   │  │ Warp 7 (32 threads) │    │
    │                          │   │  └─────────────────────┘    │
    │                          │   ├──────────────────────────────┤
    │                          │   │ Block 1                      │
    │                          │   │  ┌─────────────────────┐    │
    │                          │   │  │ 256 threads         │    │
    │                          │   │  └─────────────────────┘    │
    │                          │   ├──────────────────────────────┤
    │                          │   │ ...                          │
    │                          │   │ Block N-1                    │
    │                          │   └──────────────────────────────┘
    │                          │
    │                          └─► Each Thread:
    │                              - tid = blockIdx.x * 256 + threadIdx.x
    │                              - Compute norm for song[tid]
    │                              - Store in d_norms[tid]
    │
    ├─► cudaDeviceSynchronize() (Wait for GPU)
    │
    └─► Continue...
```

---

## 6. Data Flow Through System

```
┌───────────┐
│ CSV File  │ (Raw data: text, mixed formats)
└─────┬─────┘
      │
      ▼ Parse & Validate
┌───────────────┐
│ Valid Songs   │ (Cleaned: valid numbers, no nulls)
└───────┬───────┘
        │
        ▼ Normalize
┌───────────────┐
│ Normalized    │ (All features in [0, 1])
│ Features      │
└───────┬───────┘
        │
        ▼ Serialize
┌───────────────┐
│ Binary File   │ (Compact, fast to load)
└───────┬───────┘
        │
        ▼ Load
┌───────────────┐
│ CPU Memory    │ (std::vector<Song>)
└───────┬───────┘
        │
        ▼ Transfer
┌───────────────┐
│ GPU Memory    │ (float array, coalesced)
└───────┬───────┘
        │
        ▼ Compute
┌───────────────┐
│ Similarity    │ (Cosine similarity scores)
│ Scores        │
└───────┬───────┘
        │
        ▼ Select
┌───────────────┐
│ Top-N Results │ (Best matches)
└───────┬───────┘
        │
        ▼ Display
┌───────────────┐
│ Console       │ (Formatted output)
└───────────────┘
```

---

## 7. Build Dependency Graph

```
                     ┌──────────┐
                     │ Song.h   │
                     └─────┬────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────────┐ ┌─────────┐  ┌──────────────┐
    │DataManager.h │ │ main.cpp│  │Recommender.h │
    └──────┬───────┘ └────┬────┘  └──────┬───────┘
           │              │               │
           ▼              │               ▼
    ┌──────────────┐     │        ┌──────────────┐
    │DataManager.cpp│    │        │Recommender.cu│
    └──────┬───────┘     │        └──────┬───────┘
           │              │               │
           │              │               │
           ├──────────────┼───────────────┤
           │              │               │
           ▼              ▼               ▼
       ┌────────────────────────────────────┐
       │          g++ / nvcc                │
       │       (Compiler & Linker)          │
       └─────────────────┬──────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ recommender  │ (Executable)
                  └──────────────┘
```

---

**Diagrams Created:** October 5, 2025  
**Purpose:** Visual understanding of system architecture and data flow
