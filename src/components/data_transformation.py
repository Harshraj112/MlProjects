import sys
from dataclasses import dataclass
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        1. Numerical columns: writing_score, reading_score
           - Imputation: median
           - Scaling: StandardScaler
        2. Categorical columns: gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course
           - Imputation: most_frequent
           - Encoding: OneHotEncoder
           - Scaling: StandardScaler (with_mean=False)
        3. Combine both pipelines using ColumnTransformer
        4. Return the preprocessor object
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and categorical pipelines created successfully.")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            logging.error("Error in get_data_transformer_object: %s", e)
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully.")

            logging.info("Obtaining preprocessor object.")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing data.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object to artifacts folder
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor_obj, f)

            logging.info("Preprocessor object saved to artifacts folder.")
            logging.info("Data transformation completed successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error in initiate_data_transformation: %s", e)
            raise CustomException(e, sys)
