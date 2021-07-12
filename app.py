from flask import Flask,request,render_template,jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("home.html")
@app.route('/main',methods=['GET'])  # route to display the home page
@cross_origin()
def main():
    return render_template("index.html")
@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            CRIM = float(request.form['CRIM'])
            ZN = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])

            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = int(request.form['AGE'])
            DIS = float(request.form['DIS'])
            PTRATIO= float(request.form['PTRATIO'])
            B = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])
            is_CHAS = request.form['CHAS']
            if (is_CHAS == 'YES'):
                CHAS = 1
            else:
                CHAS = 0

            file = 'linear_regression.pickle'
            model = pickle.load(open(file, 'rb'))

            prediction = model.predict([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,PTRATIO,B,LSTAT]])

            print('PRICE IS :',prediction)
            return render_template('result.html',prediction=round(prediction[0]))
        except Exception as e:
            print('warninhg',e)
            return 'SOMETHINGS WRONG'
    else:
        return render_template('home.html')



if __name__ == "__main__":
    app.run(debug=True)
