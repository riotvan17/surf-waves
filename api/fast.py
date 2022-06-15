from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
import sys
import pandas as pd
import numpy as np
import requests
import re

sys.path.append('..')
from utils import download_model, get_data_location_wind, get_data_location_wave, get_data_tide, download_model_pkl,test_api,get_model,get_model_pkl
sys.path.pop()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(location):
    print(location)

    loc_dict = {'Oostend':['Oostend','OST', 'https://magicseaweed.com/Oostende-Surf-Report/141/'], #OMP - wind , #OST - wave
                'DePanne':['Nieuwpoort','TRG', 'https://magicseaweed.com/De-Panne-Surf-Report/4048/'], #NP7 - wind , #TRG - wave
                'Knokke':['Knokke','ZHG', 'https://magicseaweed.com/Surfers-Paradise-Surf-Report/142/']} #ZMP - wind , #ZHG - wave

    if test_api():
        tide_df = get_data_tide()
        wave_df = get_data_location_wave(loc_dict[location][1])
        wave_period = wave_df['wave_period'][0]
        wave_height = wave_df['wave_height'][0]
        tide = tide_df['tide'][0]
    else:
        data = {'wave_period':  [4.95, 4.16, 3.71],
                'wave_height': [79.0, 51.0, 57.0],
                'tide':[205.2, 206.8, 204.35 ]
                }
        wave_df = pd.DataFrame(data, index=['Oostend','DePanne','Knokke'])
        wave_period = wave_df['wave_period'][location]
        wave_height = wave_df['wave_height'][location]
        tide = wave_df['tide'][location]

    wind_df = get_data_location_wind(loc_dict[location][0])
    wind_speed = wind_df['wind_speed'][0]
    wind_direction = wind_df['wind_direction'][0]


    X_pred = pd.DataFrame([[wind_speed, wind_direction,wave_period,
                           tide]],
                          columns=['wind_speed', 'wind_direction','wave_period',
                           'tide'])
    if location == 'Oostend':
        #pipeline = download_model(model_name='Oo_model.joblib')
        #preproc = download_model_pkl(model_name='pipeline_Oo.pkl')
        pipeline = get_model('raw_data/Oo_model.joblib')
        preproc = get_model_pkl('raw_data/pipeline_Oo.pkl')
    elif location == 'DePanne':
        #pipeline = download_model(model_name='Dp_model.joblib')
        #preproc = download_model_pkl(model_name='pipeline_Dp.pkl')
        pipeline = get_model('raw_data/Dp_model.joblib')
        preproc = get_model_pkl('raw_data/pipeline_Dp.pkl')
    else:
        #pipeline = download_model(model_name='Kn_model.joblib')
        #preproc = download_model_pkl(model_name='pipeline_Kn.pkl')
        pipeline = get_model('raw_data/Kn_model.joblib')
        preproc = get_model_pkl('raw_data/pipeline_Kn.pkl')

    pipe = Pipeline([
    ('preproc', preproc),
    ('model', pipeline)])

    response = requests.get(loc_dict[location][2])
    soup = BeautifulSoup(response.content, "html.parser")
    day_next = soup.find_all('tbody')[1]
    data_block = day_next.find('small', string='Noon').parent.parent.find_all('td')
    forecast_period = float(data_block[4].text.strip().strip('s'))
    forecast_wind_speed = float(data_block[-5].find('strong').text.strip())
    forecast_wind_direction = float(re.search(r'\d+',data_block[-4]['title']).group())
    forecast_tide_low = float(day_next.find_all('tr')[-6].find_all('td')[-1].text.strip('m').strip('ft'))*100
    forecast_tide_high = float(day_next.find_all('tr')[-7].find_all('td')[-1].text.strip('m').strip('ft'))*100
    img_main = day_next.find_all('tr')[-9].find_all('img')[0]['data-src']
    img_second = day_next.find_all('tr')[-9].find_all('img')[1]['src']

    X_pred_f_low = pd.DataFrame([[forecast_wind_speed, forecast_wind_direction,forecast_period,forecast_tide_low]],
                                columns=['wind_speed', 'wind_direction','wave_period','tide'])
    X_pred_f_high = pd.DataFrame([[forecast_wind_speed, forecast_wind_direction,forecast_period,forecast_tide_high]],
                                columns=['wind_speed', 'wind_direction','wave_period','tide'], index=X_pred.index)

    rating_sc = pipe.predict(X_pred)
    print(X_pred_f_high)

    high_f_high = pipe.predict(X_pred_f_high)
    high_f_low = pipe.predict(X_pred_f_low)

    return {"rating": rating_sc[0], 'wind_speed': wind_speed, 'wind_direction': wind_direction,
            'wave_period': wave_period, 'tide': tide, 'wave_height': wave_height, 'forecast_high':high_f_high[0],
            'forecast_low':high_f_low[0], 'img_1':img_main, 'img_2':img_second}
