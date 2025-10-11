# Sample Dataset for Testing

This is a template for our music dataset. 

## Required Columns

our CSV  have these column names:
- track_id
- track_name
- artists
- danceability
- energy
- key
- loudness
- mode
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- track_genre

##from Where we Get Data

### 1. Kaggle Datasets (Free)
- **Spotify Dataset 1921-2020**: https://www.kaggle.com/yamaerenay/spotify-dataset-19212020
- **Spotify Tracks Dataset**: https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db
- **114K+ Spotify Songs**: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs

### 2. Spotify API (Requires Developer Account)
- Sign up at: https://developer.spotify.com/
- Use the Web API to fetch track features
- Audio features endpoint: https://developer.spotify.com/documentation/web-api/reference/get-audio-features

### 3. Million Song Dataset
- Academic dataset with audio features
- http://millionsongdataset.com/

## Sample CSV Format

```csv
track_id,track_name,artists,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,track_genre
5SuOikwiRyPMVoIQDJUgSV,Gen Z,Babbu,0.601,0.884,0,-3.803,1,0.182,0.0322,0.0,0.0833,0.368,133.005,dance
4qiyUfNDfud38R8iTJMld2,Lalala,Y2K,0.742,0.69,8,-6.573,0,0.0794,0.101,0.0,0.0971,0.84,130.005,dance
2tHwzyyOLoWSFqYNjeVMzj,Introspection,RAC,0.494,0.595,1,-6.461,0,0.0288,0.0221,0.905,0.166,0.0836,170.018,dance
```

## Quick Test with Small Dataset

For initial testing, create a small CSV with 10-20 songs manually or use a subset of a larger dataset:

```bash
# Extract first 100 lines from a large dataset
head -n 101 large_dataset.csv > test_dataset.csv

# Process the test dataset
./recommender --preprocess test_dataset.csv

# Try a recommendation
./recommender --song "Gen Z" -n 5
```

## Note

- Ensure there are no missing values in numerical columns
- The preprocessor will skip invalid rows automatically
- Genre names can be any string - they'll be encoded automatically
- For best results, use at least 1000+ songs
- The more data, the better the recommendations!
