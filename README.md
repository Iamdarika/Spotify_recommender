# High-Performance Music Recommendation Engine

A GPU-accelerated content-based music recommendation system built with C++, CUDA, and cuBLAS. Uses cosine similarity on audio feature vectors to find similar songs.

## 🚀 Features

- **CPU-Parallelized Preprocessing**: Uses OpenMP to efficiently process large CSV datasets
- **GPU-Accelerated Similarity**: Leverages CUDA and cuBLAS for high-performance matrix operations
- **Content-Based Filtering**: Analyzes 12 audio features (danceability, energy, tempo, etc.)
- **Binary Data Format**: Fast loading with custom serialization
- **Flexible Search**: Find songs by name or track ID
- **Genre Awareness**: Includes genre information in recommendations

## 📋 Requirements

### Software Dependencies
- **CUDA Toolkit** (10.0 or higher)
- **NVIDIA GPU** with compute capability 6.0+ (for sm_60 architecture)
- **g++** compiler with C++11 support
- **OpenMP** support (usually included with g++)

### Hardware Requirements
- NVIDIA GPU (GTX 1000 series or newer recommended)
- Multi-core CPU for preprocessing
- Sufficient RAM for dataset (depends on size)

## 🛠️ Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/mnand/Projects/SpotifyRecommendation
   ```

2. **Verify CUDA installation:**
   ```bash
   nvcc --version
   ```

3. **Build the project:**
   ```bash
   make
   ```

   This will compile all source files and create the `recommender` executable.

## 📊 Dataset Format

The input CSV should contain the following columns:
- `track_id`: Unique identifier for the track
- `track_name`: Name of the song
- `artists`: Artist name(s)
- `danceability`: How suitable for dancing (0.0 to 1.0)
- `energy`: Intensity and activity (0.0 to 1.0)
- `key`: Musical key (0 to 11)
- `loudness`: Overall loudness in dB
- `mode`: Major (1) or minor (0)
- `speechiness`: Presence of spoken words (0.0 to 1.0)
- `acousticness`: Acoustic vs electric (0.0 to 1.0)
- `instrumentalness`: Predicts instrumental tracks (0.0 to 1.0)
- `liveness`: Presence of audience (0.0 to 1.0)
- `valence`: Musical positiveness (0.0 to 1.0)
- `tempo`: Beats per minute (BPM)
- `track_genre`: Genre classification

**Example CSV structure:**
```csv
track_id,track_name,artists,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,track_genre
1,Song Name,Artist Name,0.735,0.578,5,-5.217,1,0.0461,0.514,0.0902,0.159,0.636,98.977,pop
```

## 🎯 Usage

### Step 1: Preprocess Data

Convert your CSV dataset into optimized binary format:

```bash
./recommender --preprocess <path_to_csv>
```

**Example:**
```bash
./recommender --preprocess spotify_songs.csv
```

**Output:** Creates `songs_data.bin` with normalized feature vectors

**What happens:**
- Parses CSV file in parallel using OpenMP
- Validates and cleans data
- Normalizes all features to [0, 1] range
- Encodes genres as integer IDs
- Saves to binary format for fast loading

### Step 2: Get Recommendations

#### By Song Name:
```bash
./recommender --song "Song Name" [-n N]
```

**Example:**
```bash
./recommender --song "Bohemian Rhapsody" -n 10
```

#### By Track ID:
```bash
./recommender --id "track_id_here" [-n N]
```

**Example:**
```bash
./recommender --id "3ade68b8e" -n 5
```

**Options:**
- `-n N`: Number of recommendations to return (default: 10)

## 🏗️ Architecture

### File Structure
```
SpotifyRecommendation/
├── Song.h              # Song data structure with serialization
├── DataManager.h       # Preprocessing interface
├── DataManager.cpp     # CSV parsing, normalization, OpenMP
├── Recommender.h       # GPU recommendation interface
├── Recommender.cu      # CUDA kernels, cuBLAS operations
├── main.cpp           # CLI argument parsing and orchestration
├── Makefile           # Build configuration
└── README.md          # This file
```

### Component 1: Data Preprocessor (CPU-Parallelized)

**File:** `DataManager.cpp`

**Responsibilities:**
- CSV parsing with quoted field support
- Data validation and cleaning
- Feature normalization using min-max scaling
- Genre encoding (text → integer ID)
- Binary serialization

**Parallelization:**
- Uses OpenMP `#pragma omp parallel for` for row processing
- Dynamic scheduling for load balancing
- Critical sections for genre map updates

### Component 2: Recommendation Engine (GPU-Accelerated)

**File:** `Recommender.cu`

**Responsibilities:**
- Load binary data into CPU memory
- Transfer feature vectors to GPU
- Calculate cosine similarity using cuBLAS
- Return top-N similar songs

**GPU Operations:**
1. **Dot Product Calculation** (cuBLAS SGEMV)
   - Computes dot products between query and all songs
   - Matrix-vector multiplication: `similarities = Features × Query`

2. **Norm Computation** (Custom CUDA Kernel)
   - Calculates L2 norm for each feature vector
   - Parallel execution across all songs

3. **Normalization** (Custom CUDA Kernel)
   - Divides dot products by norms to get cosine similarity
   - Formula: `cosine_sim = dot(A, B) / (norm(A) * norm(B))`

## 🧮 Similarity Calculation

**Cosine Similarity** measures the angle between two vectors:

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

Where:
- $A \cdot B$ = dot product of feature vectors
- $\|A\|$ and $\|B\|$ = L2 norms (magnitudes)
- Result ranges from -1 (opposite) to 1 (identical)

**Why Cosine Similarity?**
- Scale-invariant (focuses on direction, not magnitude)
- Efficient for high-dimensional spaces
- Well-suited for feature-based recommendations

## 🔧 Makefile Targets

```bash
make              # Build the executable
make clean        # Remove all build artifacts and data
make clean-obj    # Remove only object files
make help         # Display help message
```

## ⚙️ Configuration

### Adjusting GPU Architecture

If you have a different GPU, modify the `sm_XX` flag in the Makefile:

```makefile
NVCCFLAGS = -std=c++11 -O3 -arch=sm_60 -Xcompiler -fopenmp
                                   ^^^^^^
```

**Common values:**
- `sm_60`: GTX 1000 series (Pascal)
- `sm_70`: Tesla V100 (Volta)
- `sm_75`: RTX 2000 series (Turing)
- `sm_80`: A100, RTX 3000 series (Ampere)
- `sm_86`: RTX 3050/3060 (Ampere)
- `sm_89`: RTX 4000 series (Ada Lovelace)

Check your GPU's compute capability: https://developer.nvidia.com/cuda-gpus

### Optimization Flags

Current optimization level: `-O3` (maximum optimization)

For debugging, change to `-O0 -g`:
```makefile
CXXFLAGS = -std=c++11 -O0 -g -Wall -fopenmp
NVCCFLAGS = -std=c++11 -O0 -g -arch=sm_60 -Xcompiler -fopenmp
```

## 📈 Performance Tips

1. **Preprocessing:**
   - Set `OMP_NUM_THREADS` environment variable:
     ```bash
     export OMP_NUM_THREADS=8
     ./recommender --preprocess dataset.csv
     ```

2. **GPU Memory:**
   - For large datasets (>1M songs), consider batch processing
   - Monitor GPU memory with `nvidia-smi`

3. **Query Performance:**
   - First query may be slower (GPU warmup)
   - Subsequent queries are very fast (~ms)

## 🐛 Troubleshooting

### CUDA not found
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### OpenMP not working
Install OpenMP support:
```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# Arch Linux
sudo pacman -S openmp
```

### GPU Architecture Mismatch
Error: `no kernel image available for execution on the device`

Solution: Update `sm_XX` in Makefile to match your GPU

### Out of Memory
Reduce batch size or use GPU with more memory

## 📝 Example Output

### Preprocessing
```
Starting data preprocessing from: spotify_songs.csv
Read 114000 data rows from CSV
Parsing and validating songs...
Valid songs: 113999 out of 114000
Unique genres: 114
Normalizing features...
Writing binary data to: songs_data.bin
Preprocessing complete! Saved 113999 songs to binary file.

Genre Mapping:
  ID 0: acoustic
  ID 1: afrobeat
  ID 2: alt-rock
  ...
```

### Recommendations
```
=== RECOMMENDATION MODE ===
Loading preprocessed data from: songs_data.bin
Loaded 113999 songs and 114 genres.
Initializing GPU-accelerated recommender...
Successfully initialized with 113999 songs on GPU

Searching for song: Bohemian Rhapsody

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query Song:
  Title:   Bohemian Rhapsody
  Artist:  Queen
  Genre:   rock
  ID:      4u7EnebtmKWzUH433cf5Qv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Top 10 Recommendations:

1. "Stairway to Heaven"
   Artist: Led Zeppelin
   Genre:  rock
   ID:     5CQ30WqJwcep0pYcV4AMNc

2. "Hotel California"
   Artist: Eagles
   Genre:  rock
   ID:     40riOy7x9W7GXjyGp4pjAv
...
```

## 📚 References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [OpenMP Specification](https://www.openmp.org/)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

## 📄 License

This project is for educational purposes. Feel free to modify and extend!

## 🤝 Contributing

Suggestions for improvements:
- Implement additional similarity metrics (Euclidean, Jaccard)
- Add collaborative filtering
- Create a web interface
- Support for incremental updates
- Multi-GPU support for larger datasets

---

**Built with ❤️ using C++, CUDA, and cuBLAS**
