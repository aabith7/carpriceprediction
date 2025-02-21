from flask import Flask, render_template, request
import numpy as np
import pickle

application = Flask(__name__)
app = application


regression_model = pickle.load(open('models/regression.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods= ['GET','POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form.get('year'))
        Engine_Size = float(request.form.get('engine_size'))
        mileage = float(request.form.get('mileage'))

        new_data = standard_scaler.transform([[year,Engine_Size,mileage]])
        result = regression_model,predict(new_data)
        return render_template('form.html',result = result)

    else:
        return render_template('form.html')




if __name__ == '__main__':
    app.run(debug=True)