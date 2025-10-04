# Movie Recommendation System

This system implements two different approaches for movie recommendations:
1. Collaborative Filtering (CF)
2. Singular Value Decomposition (SVD)

## Approach
- Load rating and movie name csv file.
- Merge rating and movie name csv across movie_id.
- Splite train:test data into 70:30 ratio accordingly.
- Convert the table train-test data into matrix.
- Train the model and save necessary components.
- Evalute models on necessary matrices.
- Plot the performance data.

## Overview

### Collaborative Filtering (CF)
The CF model uses a memory-based collaborative filtering approach with the following components:
- Item-Item similarity matrix using cosine similarity
- User-Item rating matrix
- Prediction mechanism based on similarity scores

### SVD Model
The SVD-based model uses matrix factorization through the Surprise library with:
- 100 latent factors
- 20 training epochs
- Learning rate of 0.005
- Regularization parameter of 0.02commendation System

This is a Collaborative Filtering (CF) based and SVD based movie recommendation system that suggests movies to users based on their preferences and similarity with other users.

## Overview

The system uses a memory-based collaborative filtering approach with the following components:
- Item-Item similarity matrix using cosine similarity
- User-Item rating matrix
- Prediction mechanism based on similarity score

## Performance Metrics

CF model have been evaluated using standard recommendation system metrics:
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)

SVD model have been evaluated:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

## Dataset

The model is trained on the MovieLens dataset, which includes:
- User ratings for movies
- Movie metadata (titles, genres, etc.)
- User interaction data

## Usage

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Gradio web interface:
```bash
python app.py
```

3. Enter a user ID and the number of recommendations you want to receive.

## Model Details

### Collaborative Filtering Model
- **Algorithm**: Memory-based Collaborative Filtering with Item-Item similarity
- **Similarity Metric**: Cosine Similarity
- **Rating Scale**: 1-5
- **Training Data Size**: 70% of available data
- **Test Data Size**: 30% of available data

### SVD Model
- **Algorithm**: Matrix Factorization using SVD
- **Implementation**: Surprise library
- **Parameters**:
  - Number of factors: 100
  - Number of epochs: 20
  - Learning rate: 0.005
  - Regularization: 0.02
- **Rating Scale**: 1-5
- **Training Data Split**: 75% training, 25% testing

## Files Description

- `app.py`: Gradio interface for the recommendation system
- `cf_model.pkl`: Saved model components
- `requirements.txt`: List of required Python packages
- `model_dev.ipynb`: Jupyter notebook containing model development and evaluation

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Gradio
- Matplotlib
- Seaborn#

