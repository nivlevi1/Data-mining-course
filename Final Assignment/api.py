from flask import Flask, render_template, request
import pickle
import os
import numpy as np
import pandas as pd
from model_training import prepare_data_after_split

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    city = request.form['City']
    Type = request.form['type']
    room_number = float(request.form['room_number'])
    Area = float(request.form['Area'])
    num_of_images = int(request.form['num_of_images'])
    hasElevator = int(request.form['hasElevator'])
    hasParking = int(request.form['hasParking'])
    hasBars = int(request.form['hasBars'])
    hasStorage = int(request.form['hasStorage'])
    condition = request.form['condition']
    hasAirCondition = int(request.form['hasAirCondition'])
    hasBalcony = int(request.form['hasBalcony'])
    hasMamad = int(request.form['hasMamad'])
    handicapFriendly = int(request.form['handicapFriendly'])
    entrance_date = request.form['entranceDate']
    furniture = request.form['furniture']
    description = int(request.form['description'])
    floor = int(request.form['floor'])
    total_floors = int(request.form['total_floors'])
    rank = int(request.form['rank'])

    Test = pd.DataFrame({
        'City': [city],
        'type': [Type],
        'room_number': [room_number],
        'Area': [Area],
        'num_of_images': [num_of_images],
        'hasElevator': [hasElevator],
        'hasParking': [hasParking],
        'hasBars': [hasBars],
        'hasStorage': [hasStorage],
        'condition': [condition],
        'hasAirCondition': [hasAirCondition],
        'hasBalcony': [hasBalcony],
        'hasMamad': [hasMamad],
        'handicapFriendly': [handicapFriendly],
        'entranceDate': [entrance_date],
        'furniture': [furniture],
        'description': [description],
        'floor': [floor],
        'total_floors': [total_floors],
        'rank': [rank]
    })

    prediction = model.predict(Test)

    return render_template('index.html', prediction=np.round(prediction, 3))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
