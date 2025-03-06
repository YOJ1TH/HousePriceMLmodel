# flask, scikit-learn, pandas,pickle-mixin
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
data=pd.read_csv(r"C:\Users\Yojith\Downloads\Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath= request.form.get('bath')
        sqft = request.form.get('total_sqft')

        # Check if bhk and bath values are not empty
        if bhk and bath:
            try:
                bhk = float(bhk)
                bath = float(bath)
            except ValueError:
                return "Invalid input for bhk or bath. Please provide numeric values."

            input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
            prediction = pipe.predict(input_data)[0] * 1e5

            return str(np.round(prediction, 2))
        else:
            return "Please provide valid values for bhk and bath."

if __name__=="__main__":
    app.run(debug=True, port=5000)