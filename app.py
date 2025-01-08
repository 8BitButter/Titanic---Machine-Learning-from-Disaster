from src.Titanic_Project.logger import logging
from src.Titanic_Project.exception import CustomException
import sys
from src.Titanic_Project.components.data_ingestion import DataIngestion
from src.Titanic_Project.components.data_ingestion import DataIngestionConfig


if __name__=='__main__':
    logging.info("The execution has started")
    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
