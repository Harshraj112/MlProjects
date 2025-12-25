import os
import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            # log cwd and check source file
            cwd = os.getcwd()
            logging.info(f"Current working directory: {cwd}")
            data_file = os.path.join(cwd, 'notebook', 'data', 'stud.csv')
            logging.info(f"Looking for data file at: {data_file}")
            if not os.path.exists(data_file):
                logging.error(f"Data file not found: {data_file}")
                raise FileNotFoundError(f"Data file not found: {data_file}")

            df = pd.read_csv(data_file)
            logging.info('Dataset read as pandas dataframe')

            # ensure artifacts dir exists before writing any file
            artifacts_dir = os.path.dirname(self.ingestion_config.train_data_path)
            os.makedirs(artifacts_dir, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data is saved')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred in data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()