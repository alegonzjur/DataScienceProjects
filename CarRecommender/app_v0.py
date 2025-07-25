# App integration module

from flask import Flask, render_template, request, url_for
import pickle 
import numpy as np         
import pandas as pd
from datetime import datetime              


app = Flask(__name__)

# Loading model & encoder.
with open('model/model.pkl', 'rb') as f: 
    model = pickle.load(f)
with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)


fueltype = ['Hybrid','Diesel','LPG','Petrol','CNG','Plug-in Hybrid','Hydrogen']
gearbox = ['Automatic','Tiptronic','Manual','Variator']
# manufacturer = ['TOYOTA','CHEVROLET','MERCEDES-BENZ','BMW','LEXUS','HONDA','FORD', 'DAEWOO','HYUNDAI','SSANGYONG','VOLKSWAGEN','SUBARU','SUZUKI','FIAT',
#  'NISSAN','OPEL','KIA','ALFA ROMEO','MITSUBISHI','JEEP','DODGE','MAZDA','CADILLAC','VAZ','LAND ROVER','AUDI','RENAULT','SKODA','PORSCHE','CHRYSLER','JAGUAR',
#  'MINI','LINCOLN','ACURA','HUMMER','DAIHATSU','UAZ','BUICK','SCION','CITROEN','INFINITI','GMC','GAZ','PEUGEOT','VOLVO','TESLA','SEAT','ASTON MARTIN',
#  'ROVER','LAMBORGHINI','ISUZU','BENTLEY','HAVAL','ROLLS-ROYCE','MERCURY','MASERATI','SAAB','სხვა','MOSKVICH','ZAZ','FERRARI','SATURN','PONTIAC','GREATWALL']

# Main Route / Introduction
@app.route("/")
def home():
     return render_template('index.html')
 
# Price Stimator Route.
@app.route("/price", methods=['GET','POST'])
def price():
    # Recollecting data from form.
    if request.method == 'POST':
        input_data = {
            'Manufacturer' : request.form['Manufacturer'],
            'Model' : request.form['Model'],
            'Prod_year' : int(request.form['Prod_year']),
            'Mileage' : int(request.form['Mileage']),
            'Fuel_type' : request.form['Fuel_type'],
            'Gear_box_type' : request.form['Gear_box_type'],
            'Engine_volume' : float(request.form['Engine_volume']),
            'Leather_interior' : request.form['Leather_interior'],
            'Category' : request.form['Category']
            
        }
        # Feature engineering
        current_year = datetime.now().year
        input_df = pd.DataFrame([input_data])
        input_df['Car_Age'] = current_year - input_df['Prod_year']
        input_df['KmYearly'] = input_df['Mileage'] / input_df['Car_Age'].replace(0,1)
        
        # Categorical encoding.
        input_df_encoded = encoder.transform(input_df)
        
        # Prediction
        pred_log = model.predict(input_df_encoded)[0]
        pred_price = np.expm1(pred_log)
        
        return render_template('price.html', prediction=int(pred_price), data=input_data)
    
    return render_template('price.html',prediction=None)

# Car recommendation route.
@app.route('/recommender', methods=['GET','POST'])
def recommender():
    df = pd.read_csv('data/test.csv')
    if request.method == 'POST':
        budget = int(request.form['budget'])
        max_age = int(request.form['max_age'])
        fuel = request.form['fuel']
        gearbox = request.form['gearbox']
        
        current_year = datetime.now().year
        df["car_age"] = current_year - df["Prod_year"]
        
        # Doing filters
        filtered_df = df[
            (df['Price'] <= budget) &
            (df['car_age'] <= max_age) &
            (df['Fuel_type'] == fuel) &
            (df['Gear_box_type'] == gearbox)
        ]
        
        if not filtered_df.empty:
            recommendation = filtered_df.sort_values(by=['Mileage','Engine_volume']).iloc[0].to_dict()
        else:
            recommendation = None
        
        return render_template('recommendation.html', recommendation=recommendation, criteria=request.form)
    
    return render_template('recommendation.html', recommendation=None)

if __name__ == '__main__':
    app.run(debug=True)