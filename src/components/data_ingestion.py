import sys,os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logging import logging
from src.exception import CustomException
from src.components.data_transformation import data_transformation
from src.components.model_trainer import model_trainer
from dataclasses import dataclass

@dataclass
class data_ingestion_config:
    raw_path:str=os.path.join('artifacts','raw.csv')
    train_path:str=os.path.join('artifacts','train.csv')
    test_path:str=os.path.join('artifacts','test.csv')
class data_ingestion:
    def __init__(self):
        self.ingestion_config=data_ingestion_config()

    def initiate_data_ingestion(self):
        try:
            logging.info('data initiation started')
            data=pd.read_csv('C:\\Users\\Lenovo\\Downloads\\ML project\\loan_prediction\\notebook\\LoanApprovalPrediction.csv')
            
            logging.info('data ingested done')

            logging.info('making directories for storing raw data')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_path,index=False)

            logging.info('initiating train test split on original data')
            train_set,test_set=train_test_split(data,test_size=0.20,random_state=35)
            train_set.to_csv(self.ingestion_config.train_path,index=False)
            test_set.to_csv(self.ingestion_config.test_path,index=False)

            logging.info('train and test data had been uploaded to artifacts')

            return(self.ingestion_config.train_path,self.ingestion_config.test_path)
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    transformation_obj=data_transformation()
    train_arr,test_arr,_=transformation_obj.initiate_data_trasformation(train_data,test_data)

    trainer_obj=model_trainer()
    print(trainer_obj.initiate_model_trainer(train_arr,test_arr))



    

