import os
import sys
from src.Titanic_Project.exception import CustomException
from src.Titanic_Project.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.Titanic_Project.utils import read_sql_data

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ##read data from my sql database
            df = read_sql_data()
            logging.info("Reading compleated form my sql database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)


        except Exception as e:
            raise CustomException(e,sys)