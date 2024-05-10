from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from pipeline.predict_pipeline import PredictPipeline

if __name__ == "__main__":
        # Data Ingestion
        data_ingestion_obj = DataIngestion()
        raw_data_path = data_ingestion_obj.initiate_data_ingestion()

        # Data Transformation
        data_transformation_obj = DataTransformation()
        data_transformation_obj.initiate_data_transformation(raw_data_path)

        # Predict/Recomment Movie
        predict_pipeline_obj = PredictPipeline()
        movies_list = predict_pipeline_obj.recommend_movie("Batman Begins")
        print(movies_list)