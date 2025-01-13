from src.Titanic_Project.logger import logging
from src.Titanic_Project.exception import CustomException
import sys
from src.Titanic_Project.components.data_ingestion import DataIngestion
from src.Titanic_Project.components.data_transformation import DataTransformation


if __name__=='__main__':
    logging.info("The execution has started")
    try:
        # Data Ingestion
        data_ingestion=DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

        logging.info("Data Ingestion Compleated")

        # Data Transformation
        data_transformation = DataTransformation()
        train_df,test_df = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        logging.info("Data Transformation Compleated")

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
