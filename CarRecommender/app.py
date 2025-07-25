# Library imports.
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd 
import numpy as np
import lightgbm as lgb

# Initialize the Flask application.
app = Flask(__name__)

# Loading the model.
try:
    with open('./model/lgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./model/preprocessor_data.pkl', 'rb') as f:
        preprocessor_data = pickle.load(f)
    # Load preprocessor data.   
    expected_columns = preprocessor_data['expected_columns']
    mode_cylinders = preprocessor_data['mode_cylinders']
    categorical_features_info = preprocessor_data['categorical_features_info']
    lgbm_categorical_features = preprocessor_data['lgbm_categorical_features']
    
     # Creating a DF.
    reference_df_for_categories = pd.DataFrame(columns=expected_columns)
    for col in lgbm_categorical_features:
        if col in categorical_features_info:
            reference_df_for_categories[col] = pd.Categorical([], categories=categorical_features_info[col])
            
    print("Model and preprocessor data loaded successfully.")

# Loading the dataset to make recommendations.
    dataset_path = './data/Vehicle Price.csv'
    original_cars_df = pd.read_csv(dataset_path)
# Applying the same preprocessing as the model.
    if 'cylinders' in original_cars_df.columns:
        original_cars_df['cylinders'] = original_cars_df['cylinders'].replace(0.0, np.nan)
    if 'description' in original_cars_df.columns:
        original_cars_df = original_cars_df.drop('description', axis=1)
    print(f'Dataset loaded successfully with {len(original_cars_df)} records.')
except Exception as e:
    print(f"Error loading model or preprocessor data: {e}")
    model = None
    original_cars_df = None


# Define the home route.
@app.route("/")
def home():
     return render_template('index.html')
 
# Define the price estimation route.
@app.route("/price", methods=['GET', 'POST'])
def price():
    predicted_price = None 
    error_message = None
    # If the model is not loaded, return an error message.
    if model is None:
        error_message = "Model is not loaded. Please try again later."
        return render_template('price.html', prediction=predicted_price, error=error_message)
    
    if request.method == 'POST':
        try:
            # Collecting data from the form.
            car_data = {
                'make' : request.form['make'],
                'model' : request.form['model'],
                'year' : int(request.form['year']),
                'engine' : float(request.form['engine']),
                'cylinders' : int(request.form['cylinders']),
                'fuel' : request.form['fuel'],
                'mileage' : int(request.form['mileage']),
                'transmission' : request.form['transmission'],
                'trim' : request.form['trim'],
                'body' : request.form['body'],
                'doors' : int(request.form['doors']),
                'exterior_color' : request.form['exterior_color'],
                'interior_color' : request.form['interior_color'],
                'drivetrain' : request.form['drivetrain']
            }
            # Creating a DataFrame from the input data.
            input_df = pd.DataFrame([car_data])
            # Checking that the dataframe has the expected columns.
            try:
                input_df = input_df[expected_columns]
            except KeyError as e:
                raise KeyError(f"Missing expected column: {e}")
            # Impute cylinders if not provided.
            if 'cylinders' in input_df.columns and input_df['cylinders'].isnull().any():
                input_df['cylinders'] = input_df['cylinders'].replace(0, np.nan)
                input_df['cylinders'].fillna(mode_cylinders, inplace=True)
                input_df['cylinders'] = input_df['cylinders'].astype(int)
            # Ensure categorical columns are treated as categories.
            for col in lgbm_categorical_features:
                if col in input_df.columns:
                    if col in reference_df_for_categories.columns:
                        input_df[col] = pd.Categorical(input_df[col], categories=reference_df_for_categories[col].cat.categories)
                    else:
                        pass
            # Making predictions.
            predicted_price = model.predict(input_df)[0]
        except ValueError as e:
            error_message = f"Invalid input data: {e}"
        except Exception as e:
            error_message = f"An error occurred while making the prediction: {e}"
    return render_template('price.html', predicted_price=predicted_price, error_message=error_message)

# Recommendation route. In works.
# @app.route("/recommend_car", methods=['GET','POST'])
# def recommend_car():
#     recommend_cars = []
#     error_message = None
#     # If the model is not loaded, return an error message.
#     if original_cars_df is None:
#         error_message = 'Internal error. Car catalog is not available.'
#         return render_template('recommendation.html', recommend_cars=recommend_cars, error_message=error_message)
#     # Collecting information for the car.
#     if request.method == 'POST':
#         try:
#             budget = float(request.form['max_budget'])
#             age = 
             
            

# Debugging.
if __name__ == "__main__":
    app.run(debug=True)