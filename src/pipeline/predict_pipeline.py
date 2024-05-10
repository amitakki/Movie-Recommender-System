import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:



    def recommend_movie(self, movie_name) -> list:
        '''
        This function recommends list of movies similar to the movie_name
        '''
        try:
            # silimarity and preprocessed data file path 
            similarity_path = os.path.join("artifacts", "similarity.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocesor.pkl")

            logging.info("Before loading pkl files")

            # read both the pickle files
            similarity = load_object(file_path=similarity_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("After loading pkl files")
             
            # convert the preprocessed movie data to dataframe
            movies_df = pd.DataFrame(preprocessor)

            # find the index of the movie_name in processed dataframe
            movie_index = movies_df[movies_df['title'] == movie_name].index[0]
            # find the similar vector distances to the movie_name index
            distances = similarity[movie_index]
            # sort the vector with similar movie_distances and pick the first 5 movies
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6] 

            # fetch the similar movie names using the index
            recommended_movies = []
            for movie in movies_list:
                recommended_movies.append(movies_df.iloc[movie[0]].title)

            return recommended_movies

        except Exception as e:
            raise CustomException(e, sys)

