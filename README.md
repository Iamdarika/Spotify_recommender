# Spotify-Scale Music Recommendation Engine (HPC Project)

Team B7:

- Darika N S (OpenMP + Integration)
- Arunavarshini N (Preprocessing with OpenMP)
- Jonnala Thanishka (MPI Benchmarking)
- Killana Pavan Kumar (CUDA Collaborative Filtering)

### Description
A high-performance music recommendation engine built on:
- **Matrix Factorization** (CUDA for acceleration)
- **Collaborative Filtering** (CUDA for recommendation)
- **Parallel Preprocessing** (OpenMP)
- **Benchmarking** (MPI)

### Project Structure
- **data/** → Spotify dataset (local only, excluded via `.gitignore`)
- **src/** → Source code
- **docs/** → Reports, diagrams, documentation

### Dataset (local only, not uploaded to GitHub)
Our dataset has the following columns:
- `track_id`, `artist`, `track_name`, `genre`
- `tempo`, `valence`, `energy`, `danceability`, `acousticness`
- `popularity`, `year`, `duration_ms`
