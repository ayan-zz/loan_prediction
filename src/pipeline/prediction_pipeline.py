import sys,os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
              raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, 
                 Loan_ID : str,
                 Gender :str,
                 Married: str,
                 Dependents: str,
                 Education: str,
                 Self_Employed: str,
                 ApplicantIncome: float,
                 CoapplicantIncome: float,
                 LoanAmount: float,
                 Loan_Amount_Term: float,
                 Credit_History: float,
                 Property_Area: str
                 ):
                
                self.Loan_ID =Loan_ID
                self.Gender = Gender
                self.Married = Married
                self.Dependents = Dependents
                self.Education=Education
                self.Self_Employed=Self_Employed
                self.ApplicantIncome=ApplicantIncome
                self.CoapplicantIncome = CoapplicantIncome
                self.LoanAmount= LoanAmount
                self.Loan_Amount_Term= Loan_Amount_Term
                self.Credit_History=Credit_History
                self.Property_Area=Property_Area
                
    def get_data_as_dataframe(self):
          try:
                custom_data_input_dict={
                      "Loan_ID":[self.Loan_ID], "Gender":[self.Gender],
                      "Married":[self.Married],"Dependents":[self.Dependents],
                      "Education":[self.Education],"Self_Employed":[self.Self_Employed],
                      "ApplicantIncome":[self.ApplicantIncome],"CoapplicantIncome":[self.CoapplicantIncome],"CoapplicantIncome":[self.CoapplicantIncome],
                      "LoanAmount":[self.LoanAmount],"Loan_Amount_Term":[self.Loan_Amount_Term],"Credit_History":[self.Credit_History],
                      "Property_Area":[self.Property_Area]          
                }
                return pd.DataFrame(custom_data_input_dict)
          
          except Exception as e:
                raise CustomException(e,sys)
          