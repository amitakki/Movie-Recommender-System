# Movie Recommendation System

This project implements a movie recommendation system using machine learning techniques. The recommendation system suggests movies to users based on similarities with other users.

## Introduction

Movie recommendation systems are widely used in various online platforms such as streaming services, e-commerce websites, and social media platforms to provide personalized recommendations to users. These systems help users discover new content that aligns with their interests and preferences, thereby enhancing user experience and engagement.

This project aims to develop a movie recommendation system using machine learning algorithms. The system analyzes user preferences and movie features to generate personalized recommendations for users. It leverages content-based filtering approach to provide accurate and diverse recommendations.

## Features

- **User-Based Collaborative Filtering**: The system identifies similar users based on their movie ratings and recommends movies that similar users have enjoyed.
- **Item-Based Collaborative Filtering**: The system identifies similar movies based on their features (genres, actors, directors, etc.) and recommends movies that are similar to those that the user has liked.
- **Matrix Factorization**: The system applies matrix factorization techniques such as Singular Value Decomposition (SVD) or Alternating Least Squares (ALS) to factorize the user-item interaction matrix and generate recommendations.
- **Hybrid Approaches**: The system combines multiple recommendation algorithms to provide more accurate and diverse recommendations, taking advantage of both collaborative filtering and content-based filtering.

## Dataset

The project utilizes the [TMDB Movie dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) as the dataset for training and evaluation. The dataset contains full credits for both the cast and the crew, rather than just the first three actors.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Nltk

## Usage

1. Clone the repository: ```git clone https://github.com/username/movie-recommendation-system.git```

2. Install the required dependencies: ```pip install -r requirements.txt```

3. Download and preprocess the Movie and Credits dataset.

4. Create a Similarity Matrix using Cosine Similarity metrics
   
5. Evaluate the performance of the recommendation system.

7. Use the similarity matrix to generate recommendations for users.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- Amit Akki (@amitakki)
  
## Acknowledgements

- This dataset was generated from The Movie Database API. This product uses the TMDb API but is not endorsed or certified by TMDb.


