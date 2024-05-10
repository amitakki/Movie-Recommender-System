import sys
import os
import ast

from dataclasses import dataclass
from nltk.stem.porter import PorterStemmer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DataTransformationConfig:
    # preprocessed pickle file path
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocesor.pkl")
    # similarity pickle file path
    similarity_obj_file_path = os.path.join('artifacts', 'similarity.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()

    def initiate_data_transformation(self, raw_data_path):
        try:
            # read the movie data csv file
            movies_df = pd.read_csv(raw_data_path)

            logging.info("Data Set read")

            # drop rows with na values
            movies_df.dropna(inplace=True)

            # drop duplicates if any
            movies_df.drop_duplicates(inplace=True)

            # convert the 'genres' json text to list
            movies_df['genres'] = movies_df['genres'].apply(self.convert)
            # convert the 'keywords' json text to list
            movies_df['keywords'] = movies_df['keywords'].apply(self.convert)
            # convert the 'cast' json text to list
            movies_df['cast'] = movies_df['cast'].apply(self.convert,count=3)

            # fetch the crew with job as 'director'
            movies_df['crew'] = movies_df['crew'].apply(self.fetch_director)

            # split the words in 'overview' column
            movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())

            # remove the space between words in 'genres', 'keywords', 'cast' and 'crew'
            movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
            movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
            movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
            movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

            # create a new column 'tags' that combines all the features above
            movies_df['tags'] = movies_df['overview'] + movies_df['genres'] +\
                movies_df['keywords'] + movies_df['cast'] + movies_df['crew']
            
            # select only the 'movie_id', 'title', and 'tags' features
            movies_df = movies_df[['movie_id', 'title', 'tags']]

            # create a list for words in tags joining them by space 
            movies_df['tags'] = movies_df['tags'].apply(lambda x:" ".join(x))

            # convert all letters to lower case in tags column
            movies_df['tags'] = movies_df['tags'].apply(lambda x:x.lower())

            # stem the english words which are not useful for recommendation system
            movies_df['tags'] = movies_df['tags'].apply(self.stemmer)

            # save the final processed dataframe to a pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = movies_df.to_dict()
            )

            # create a vector with common tags for 5000 features
            cv = CountVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(movies_df['tags']).toarray()

            # find the cosine_similarity matrix for the above vector
            similarity = cosine_similarity(vectors)

            # save the similarity matrix to a pickle file
            save_object(file_path = self.data_transformation_config.similarity_obj_file_path,
                        obj = similarity
                        )
        except Exception as e:
            raise CustomException


    # function to convert json text to list
    def convert(self, genres_list, count=-1):
        genres = []
        for index, gen in enumerate(ast.literal_eval(genres_list)):
           if count != -1 and index >= count:
                break
           genres.append(gen['name'])
        return genres
    
    # function to fetch director from crew list
    def fetch_director(self, crew):
        director_list=[]
        for item in ast.literal_eval(crew):
            if item['job'] == 'Director':
                director_list.append(item['name'])
                break
        return director_list
    
    # function to stem connnector words from tags
    def stemmer(self, text):
        ps = PorterStemmer()
        y=[]
        for word in text.split():
            y.append(ps.stem(word))
        
        return " ".join(y)