import os
import sys
from src.Titanic_Project.exception import CustomException
from src.Titanic_Project.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.Titanic_Project.utils import read_sql_data
from sklearn.model_selection import StratifiedShuffleSplit

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Read data from MySQL database
            df = read_sql_data()
            logging.info("Reading completed from MySQL database")
            
            # Create directory for artifacts if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Implement Stratified Shuffle Split
            logging.info("Starting Stratified Shuffle Split")
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            
            for train_index, test_index in split.split(df, df[['Survived', 'Pclass', 'Sex']]):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]

            # Save train and test sets to CSV files
            strat_train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            strat_test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
