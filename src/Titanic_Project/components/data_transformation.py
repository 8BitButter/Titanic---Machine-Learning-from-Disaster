import os
import sys
from src.Titanic_Project.exception import CustomException
from src.Titanic_Project.logger import logging

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib  # For saving the pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_pipeline(self):
        '''
        This function creates the data transformation pipeline.
        '''
        try:
            pipeline = Pipeline([
                ("age_imputer", AgeImputer()),
                ("feature_encoder", FeatureEncoder()),
                ("feature_dropper", FeatureDropper())
            ])
            logging.info("Pipeline created successfully.")
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function applies the transformation pipeline to the train and test datasets.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test files loaded successfully.")

            pipeline = self.get_data_transformer_pipeline()

            train_df_transformed = pipeline.fit_transform(train_df)
            test_df_transformed = pipeline.transform(test_df)

            # Save the pipeline
            joblib.dump(pipeline, self.data_transformation_config.preprocessor_obj_file_path)
            logging.info(f"Pipeline saved at {self.data_transformation_config.preprocessor_obj_file_path}")
            
            return train_df_transformed, test_df_transformed
        except Exception as e:
            raise CustomException(e, sys)


class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy="mean")
        self.imputer.fit(X[['Age']])
        return self

    def transform(self, X):
        X = X.copy()
        X['Age'] = self.imputer.transform(X[['Age']])
        return X


class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.embarked_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.sex_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        self.embarked_encoder.fit(X[['Embarked']])
        self.sex_encoder.fit(X[['Sex']])
        return self

    def transform(self, X):
        X = X.copy()

        embarked_encoded = self.embarked_encoder.transform(X[['Embarked']])
        sex_encoded = self.sex_encoder.transform(X[['Sex']])

        embarked_columns = self.embarked_encoder.get_feature_names_out(['Embarked'])
        sex_columns = self.sex_encoder.get_feature_names_out(['Sex'])

        for i, col in enumerate(embarked_columns):
            X[col] = embarked_encoded[:, i]

        for i, col in enumerate(sex_columns):
            X[col] = sex_encoded[:, i]

        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=["Embarked", "Name", "Ticket", "Cabin", "Sex"], errors="ignore")
