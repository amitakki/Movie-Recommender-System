import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'movies_data.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Read the Dataset as dataframe")

        try:
            # read movies and credits data set
            movies_df = pd.read_csv("notebook\data\\tmdb_5000_movies.csv")
            credits_df = pd.read_csv("notebook\data\\tmdb_5000_credits.csv")

            # merge both data sets on column 'title'
            movies_df = movies_df.merge(credits_df, on='title')

            # choose the features that are relevant and required for recommending system
            movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

            # create the artifcats folder to save the above movies dataframe
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            # write the data to a file in artifacts
            movies_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data Ingestion is complete")

            # return the merged and saved movie data file path
            return self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(e, sys)