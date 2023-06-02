from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('results.html')
    
    else:
        data=CustomData(
            Loan_ID=request.form.get('Loan_ID'),
            Gender=(request.form.get('Gender')),
            Married=(request.form.get('Married')),
            Education=(request.form.get('Education')),
            Dependents=(request.form.get('Dependents')),
            Property_Area=(request.form.get('Property_Area')),
            Self_Employed=(request.form.get('Self_Employed')),
            ApplicantIncome=float(request.form.get('ApplicantIncome')),
            CoapplicantIncome=float(request.form.get('CoapplicantIncome')),
            LoanAmount=float(request.form.get('LoanAmount')),
            Loan_Amount_Term=float(request.form.get('Loan_Amount_Term')),
            Credit_History=float(request.form.get('Credit_History'))
           
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)

        categories={0:'NOT APPROVED',1:'APPROVED'}
        
        results = categories[result[0]]
        

        return render_template("results.html",results=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
    
    
     
