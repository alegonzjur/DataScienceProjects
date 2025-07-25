# Libraries.
from fastapi import FastAPI
from pydantic import BaseModel 
import pandas as pd    
import numpy as np    
import pickle 

# App 
app = FastAPI(title='Bot Detector')

# Loading the model.
with open('./models/model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Designing a class for the user data inserted.
class UserData(BaseModel):
    friends_count: int
    followers_count: int
    statuses_count: int
    lifetime_account: int
    followers_to_friends_ratio: float
    tweets_per_day: float
    description_length: int
    
# Routes.
@app.post('/predict/')
def predict_user(data: UserData):
    # Temporal dataframe.
    df_user = pd.DataFrame([data.model_dump()])
    # Replace inf for NaN, after for 0.
    df_user = df_user.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Predict.
    pred = model.predict(df_user)
    if pred[0] == 1:
        return 'The user inserted is a Bot.'
    else: 
        return 'The user inserted is a Human.'