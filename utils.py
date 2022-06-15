import os
import joblib
import pandas as pd
from google.cloud import storage
import requests
import datetime
import numpy as np
import pickle
from io import BytesIO
import math

TOKEN_2 = os.environ['TOKEN_2']
API_USER = os.environ['API_USER']
API_PASS = os.environ['API_PASS']
MODEL_URL = os.environ['MODEL_URL']

def api_connect():
    url = 'https://api.meetnetvlaamsebanken.be/Token'
    body = {
    'grant_type':'password',
    'username': API_USER,
    'password': API_PASS
    }
    response = requests.post(url, data=body)
    return response

def init_api_connection():

    response = api_connect()
    return response.json()['access_token']

def test_api():

    return api_connect().status_code == 200

def get_data_location_wave(location):

    TK = init_api_connection()
    token = {'authorization':f'Bearer {TK}'}

    r= requests.get('https://api.meetnetvlaamsebanken.be/V2/catalog', headers=token)
    cat = r.json()

    list_obj = pd.DataFrame.from_dict(cat['AvailableData'])[pd.DataFrame.from_dict(cat['AvailableData'])['Location'] == location]['ID'].to_list()
    param_list = pd.DataFrame.from_dict(cat['AvailableData'])[pd.DataFrame.from_dict(cat['AvailableData'])['Location'] == location]['Parameter'].to_list()
    col_name = pd.DataFrame.from_dict(cat['Parameters'])[pd.DataFrame.from_dict(cat['Parameters'])['ID'].isin(param_list)]['Name'].str[1].str['Message'].to_list()

    url = 'https://api.meetnetvlaamsebanken.be/V2/currentData'
    cols_used = dict(zip(list_obj, col_name))

    combined_df = []
    for i in list_obj:
        params = {
            "IDs": i}
        r = requests.post(url, data=params,  headers=token)
        repl = r.json()
        if repl == []:
            repl = [{'ID': np.nan, 'Timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M+00:00"), 'Value': np.nan}]
        test = pd.DataFrame(repl).set_index('Timestamp')
        test.rename(columns={'Value': i},inplace=True, errors='raise')
        combined_df.append(test)
    combined_df = pd.concat(combined_df)

    combined_df = combined_df.rename(columns = cols_used)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df = combined_df[['Wave height','Average wave period']]
    combined_df.columns = ['wave_height', 'wave_period']
    return combined_df.mean().to_frame().T

def get_data_tide():

    TK = init_api_connection()
    token = {'authorization':f'Bearer {TK}'}

    r= requests.get('https://api.meetnetvlaamsebanken.be/V2/catalog', headers=token)
    cat = r.json()

    url = 'https://api.meetnetvlaamsebanken.be/V2/currentData'

    list_tide_loc = pd.DataFrame.from_dict(cat['AvailableData'])[pd.DataFrame.from_dict(cat['AvailableData'])['Parameter'] == 'WS5']["Location"].to_list()
    temp_df = pd.DataFrame(columns=['Timestamp'])
    temp_df = temp_df.set_index('Timestamp')

    for i in list_tide_loc:
        list_obj = pd.DataFrame.from_dict(cat['AvailableData'])[pd.DataFrame.from_dict(cat['AvailableData'])['Location'] == i]['ID'].to_list()
        param_list = pd.DataFrame.from_dict(cat['AvailableData'])[pd.DataFrame.from_dict(cat['AvailableData'])['Location'] == i]['Parameter'].to_list()
        col_name = pd.DataFrame.from_dict(cat['Parameters'])[pd.DataFrame.from_dict(cat['Parameters'])['ID'].isin(param_list)]['Name'].str[1].str['Message'].to_list()
        cols_used = dict(zip(list_obj, col_name))

        combined_df = []
        for j in list_obj:
            params = {
                "IDs": j}
            r = requests.post(url, data=params,  headers=token)
            repl = r.json()
            if repl == []:
                repl = [{'ID': np.nan, 'Timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M+00:00"), 'Value': np.nan}]
            test = pd.DataFrame(repl).set_index('Timestamp')
            test.rename(columns={'Value': j},inplace=True, errors='raise')
            combined_df.append(test)
        combined_df = pd.concat(combined_df)

        combined_df = combined_df.rename(columns = cols_used)
        combined_df.index = pd.to_datetime(combined_df.index)

        combined_df.rename(columns={'tide': i}, inplace=True)
        temp_df = pd.concat([combined_df,temp_df])

    temp_df = temp_df.dropna(axis = 0, how = 'all')
    temp_df = temp_df.groupby('Timestamp',dropna=False).mean()
    temp_df['tide'] = round(temp_df.mean(axis=1),1)

    return temp_df[['tide']].mean().to_frame().T

def get_data_location_wind(location):

    url = f'http://api.worldweatheronline.com/premium/v1/weather.ashx?key={TOKEN_2}&q={location}&format=json&num_of_days=1'

    r= requests.get(url)
    res = r.json()

    test = pd.DataFrame(res['data']['current_condition'][0])[['windspeedKmph','winddirDegree']]
    test = test.astype({"windspeedKmph": 'float64',
                "winddirDegree": 'float64'})
    test['Timestamp'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M+00:00")
    test = test.set_index('Timestamp')
    test.index = pd.to_datetime(test.index, utc=True)
    test.columns = ['wind_speed', 'wind_direction']

    return test

def get_historical_data_all(period_from_y, period_to_y, location, max_date='2022-05-29'):

    TK = init_api_connection()
    token = {'authorization':f'Bearer {TK}'}

    r= requests.get('https://api.meetnetvlaamsebanken.be/V2/catalog', headers=token)
    cat = r.json()

    list_obj = pd.DataFrame.from_dict(cat['AvailableData'])[pd.DataFrame.from_dict(cat['AvailableData'])['Location'] == location]['ID'].to_list()
    param_list = pd.DataFrame.from_dict(cat['AvailableData'])[pd.DataFrame.from_dict(cat['AvailableData'])['Location'] == location]['Parameter'].to_list()
    col_name = pd.DataFrame.from_dict(cat['Parameters'])[pd.DataFrame.from_dict(cat['Parameters'])['ID'].isin(param_list)]['Name'].str[1].str['Message'].to_list()
    url = 'https://api.meetnetvlaamsebanken.be/V2/getData'
    df_full = pd.DataFrame()
    cols_used = dict(zip(list_obj, col_name))

    date_s = datetime.date(period_from_y,1,1)
    max_date_dt = datetime.datetime.strptime(max_date, '%Y-%m-%d').date()
    date_m = min(datetime.date(period_to_y,1,1),max_date_dt)
    loops = -(-(date_m - date_s).days//365)

    for j in range(loops):
        date_e = min(date_s + datetime.timedelta(days=365),max_date_dt)
        run = True
        params = {
            "StartTime": f"{date_s}T00:00:00.000Z",
            "EndTime": f"{date_e}T00:00:00.000Z",
            "IDs": list_obj[0]}
        r = requests.post(url, data=params,  headers=token)
        repl = r.json()['Values'][0]['Values']
        if repl == []:
            run = False
            date_s = date_e + datetime.timedelta(days=1)
            continue
        df = pd.DataFrame(repl).set_index('Timestamp')
        df.rename(columns={'Value': list_obj[0]},inplace=True, errors='raise')

        for i in list_obj[1:]:
            if run == False:
                break
            url = 'https://api.meetnetvlaamsebanken.be/V2/getData'
            params = {
            "StartTime": f"{date_s}T00:00:00.000Z",
            "EndTime": f"{date_e}T00:00:00.000Z",
            "IDs": i}
            r = requests.post(url, data=params,  headers=token)
            repl = r.json()['Values'][0]['Values']
            if repl == []:
                continue
            test = pd.DataFrame(repl).set_index('Timestamp')
            test.rename(columns={'Value': i},inplace=True, errors='raise')
            df[i] = test

        date_s = date_e + datetime.timedelta(days=1)
        df_full = df_full.append(df)

    df_full = df_full.rename(columns = cols_used)
    df_full.index = pd.to_datetime(df_full.index, utc=True)

    return df_full


def get_historical_data_wave(period_from_y, period_to_y, location, max_date='2022-05-29'):

    df_full = get_historical_data_all(period_from_y, period_to_y, location, max_date='2022-05-29')
    df_full = df_full[['Wave height','Average wave period']]
    df_full.columns = ['wave_height', 'wave_period']

    return df_full

def get_historical_data_tide():

    TK = init_api_connection()
    token = {'authorization':f'Bearer {TK}'}

    r= requests.get('https://api.meetnetvlaamsebanken.be/V2/catalog', headers=token)
    cat = r.json()

    list_tide_loc = pd.DataFrame.from_dict(cat['AvailableData'])[pd.DataFrame.from_dict(cat['AvailableData'])['Parameter'] == 'WS5']["Location"].to_list()
    temp_df = pd.DataFrame(columns=['Timestamp'])
    temp_df = temp_df.set_index('Timestamp')

    for i in list_tide_loc:
        print(f'{list_tide_loc.index(i)+1}/{len(list_tide_loc)}')
        temp = get_historical_data_all(2000, 2022, i)[['Tide TAW']]
        temp.rename(columns={'tide': i}, inplace=True)
        temp_df = pd.concat([temp,temp_df])

    temp_df = temp_df.dropna(axis = 0, how = 'all')
    temp_df = temp_df.groupby('Timestamp',dropna=False).mean()
    temp_df['tide'] = round(temp_df.mean(axis=1),1)

    return temp_df[['tide']]


def get_historical_data_wind(location='Knokke'):
    start_date_str = '2022-05-29'
    max_end_date_str = '2009-01-01'
    date_1 = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    date_end_f = datetime.datetime.strptime(max_end_date_str, "%Y-%m-%d")
    loops = int((date_1-date_end_f).days/35)
    combined_df = []

    for i in range(loops):
        end_date = (date_1 - datetime.timedelta(days=35)*(i+1)).strftime("%Y-%m-%d")
        start_date = (date_1 - datetime.timedelta(days=35)*i).strftime("%Y-%m-%d")
        url = f'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key={TOKEN_2}&q={location}&format=json&date={end_date}&enddate={start_date}&tp=1'
        r= requests.get(url)
        res = r.json()
        date_list = [res['data']['weather'][i]['date'] for i in range(len(res['data']['weather']))]
        for i in range(len(res['data']['weather'])):
            test = pd.DataFrame(res['data']['weather'][i]['hourly'])[['time','windspeedKmph','winddirDegree']]
            test['date'] = date_list[i]
            test['time'] = (test['time'].astype(int)/100).astype(int)
            test['Timestamp']= test.apply(lambda x:datetime.datetime.strptime("{0} {1}".format(x['date'], x['time']), "%Y-%m-%d %H"),axis=1)
            test = test.set_index('Timestamp')
            test.index = pd.to_datetime(test.index,utc=True)
            test = test.drop(columns=['time','date'])
            combined_df.append(test)
    combined_df = pd.concat(combined_df)
    combined_df.columns = ['wind_speed', 'wind_direction']
    return combined_df


def download_model(model_name='model.joblib'):

    url = MODEL_URL
    r = requests.get(url+model_name)
    r_cont = BytesIO(r.content)
    print(f"{model_name} => pipeline downloaded from storage")
    model = joblib.load(r_cont)


    return model

def download_model_pkl(model_name='pipeline.pkl'):

    url = MODEL_URL
    r = requests.get(url+model_name)
    r_cont = BytesIO(r.content)
    print(f"{model_name} => pkl downloaded from storage")
    model = pickle.load(r_cont)
    return model

def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

def get_model_pkl(path):
    pipeline = pickle.load(open(path,"rb"))
    return pipeline

def tide_cat(num):
    if num < 200:
        return 1
    elif num >= 200 and num < 300:
        return 2
    else:
        return 3
def trans_func(df):
    return df.apply(lambda x: [tide_cat(num) for num in x])
def cos_list(df):
    return np.cos(df.apply(lambda x: [2 * math.pi * float(num) / 360  for num in x]))
def sin_list(df):
    return np.sin(df.apply(lambda x: [2 * math.pi * float(num) / 360  for num in x]))


if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    test_api()
