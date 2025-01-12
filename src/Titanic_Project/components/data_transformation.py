import os
import sys
from src.Titanic_Project.exception import CustomException
from src.Titanic_Project.logger import logging

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            
            pass

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            pass

        
        except Exception as e:
            raise CustomException(sys,e)