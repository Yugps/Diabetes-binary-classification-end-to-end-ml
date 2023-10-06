from flask import Flask,render_template,request 
import pickle 
import sklearn
import pandas as pd

scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('logistic_model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict_datapoint():
    result='' # given this here other wise it gives local and global variable error 'mentioned before assignment'
    if request.method=='POST':
        pregnancies=float(request.form.get('Pregnancies'))
        glucose=float(request.form.get('Glucose'))
        bloodpressure=float(request.form.get('BloodPressure'))     
        skinthickness=float(request.form.get('SkinThickness'))
        insulin=float(request.form.get('Insulin'))
        bmi=float(request.form.get('BMI'))
        diabetespedigreefunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))
        input_params=pd.DataFrame([pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,diabetespedigreefunction,Age]).T
        scaler_params=scaler.transform(input_params)
        pred=model.predict(scaler_params)
        if pred[0]==0:
            result='Not a diabetic'
        elif pred[0]==1:
            result='Diabetic'
    return render_template('home.html',result=result)
if __name__=='__main__':
    app.run(host='0.0.0.0')
        