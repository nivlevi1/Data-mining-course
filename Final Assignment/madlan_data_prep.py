import requests
import pandas as pd
import os.path
import re
import time
import numpy as np 
import math

def prepare_data(data):
    import requests
    import pandas as pd
    import os.path
    import re
    import time
    import numpy as np 
    import math
    from datetime import datetime, timedelta
    #strip the columns headers
    data.columns = data.columns.str.rstrip() #Delete white space in the headers
    
    #Fix the data collection mistake
    data['City'] = data['City'].replace('נהרייה', 'נהריה')
    data['description'].fillna('', inplace=True)
    data['number_in_street'] = data['number_in_street'].replace('None', np.nan) # Replace 'None' values with NaN
    data['number_in_street'] = pd.to_numeric(data['number_in_street'], errors='coerce')  # Convert the column to numeric, NaN values will be retained
    data['number_in_street'] = data['number_in_street'].astype(float).astype('Int64')     # Convert the column to integer
    data['city_area'] = data['city_area'].replace('nan', np.nan)
    data['city_area'] = data['city_area'].replace([None], np.nan)
    data = data.dropna(subset=['city_area'])    
    
    # Dropping all notes which are not numbers or dot.
    def extract_name(string): #because use use that for Area too, so i chagnge the name of the function to more general name
        pattern = r'[^\d.]'
        cleaned_string = re.sub(pattern, '', string) 
        return float(cleaned_string) if cleaned_string else None  

    data['price'] = data['price'].astype(str).apply(extract_name)
    data['room_number'] = data['room_number'].astype(str).apply(extract_name)
    data['Area'] = data['Area'].astype(str).apply(extract_name)
    
    # Drpoing all rows with missimg values in the 'price' column    
    data = data.dropna(subset=['price'])
    
    #Replace the 'floor_out_of' column
    def add_floor(row):
        if pd.notnull(row['floor_out_of']):
            if 'קומה' in row['floor_out_of']:
                return int(row['floor_out_of'].split()[1])
            elif 'קומת קרקע' in row['floor_out_of']:
                return 0
            elif 'קומת מרתף' in row['floor_out_of']:
                return -1
        return np.nan

    def add_total_floors(row):
        if pd.notnull(row['floor_out_of']):
            if 'מתוך' in row['floor_out_of']:
                return int(row['floor_out_of'].split()[-1])
        return np.nan

    data.loc[:, 'floor'] = data.apply(add_floor, axis=1)
    data.loc[:, 'total_floors'] = data.apply(add_total_floors, axis=1)
    
    #Replace all the Binary values :
    def booli(value):
        if pd.isna(value) or value == "NA" or value == "inf":
            return 0
        elif value == 'כן' or value == 1 or value == 'yes' or value == 'נגיש לנכים' or value == 'יש ממ"ד' or value == 'יש מרפסת' or value == 'יש מיזוג אויר' or value == 'יש מעלית' or value == 'יש חנייה' or value == 'יש מחסן' or value == 'יש סורגים':
            return 1
        else:
            return 0

    data.loc[:, 'hasElevator'] = data['hasElevator'].fillna(0).apply(booli).astype(int)
    data.loc[:, 'hasBars'] = data['hasBars'].fillna(0).apply(booli).astype(int)
    data.loc[:, 'hasParking'] = data['hasParking'].fillna(0).apply(booli).astype(int)
    data.loc[:, 'hasStorage'] = data['hasStorage'].fillna(0).apply(booli).astype(int)
    data.loc[:, 'hasAirCondition'] = data['hasAirCondition'].fillna(0).apply(booli).astype(int)
    data.loc[:, 'hasBalcony'] = data['hasBalcony'].fillna(0).apply(booli).astype(int)
    data.loc[:, 'hasMamad'] = data['hasMamad'].fillna(0).apply(booli).astype(int)
    data.loc[:, 'handicapFriendly'] = data['handicapFriendly'].fillna(0).apply(booli).astype(int)
    
    #Replace the entranceDate column to categorical.
    today = datetime.today()

    def update_value(value):
        if isinstance(value, datetime):
            time_difference = today - value
            months_difference = time_difference.days // 30

            if months_difference < 6:
                return "less_than_6_months"
            elif 6 <= months_difference < 12:
                return "months_6_12"
            else:
                return "above_year"
        elif isinstance(value, str):
            if "גמיש" in value or "גמיש " in value:
                return "flexible"
            elif "מיידי" in value:
                return "less_than_6_months"
            elif "לא צויין" in value:
                return "not defined"

        return value
    data['entranceDate'] = data['entranceDate'].apply(update_value)
    data = data.drop('floor_out_of', axis=1) #Not relevant
    data = data.drop('publishedDays', axis=1) #Too much Nans
    data = data.drop('number_in_street', axis=1) # Could make mistake because the model will think these numbers have real value.
    data = data.drop('Street', axis=1) #We cant apply one hot encoding to a lot to streets & multicollinearity with the street_area.

    
    #Remove row without with_missing_data :
    data = data.dropna(subset=['condition'])
    data = data.dropna(subset=['floor'])
    data = data.dropna(subset=['room_number'])

    #Descripition = number of words (We dont use NLP):
    data['description'] = data['description'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

    


    
    #Label encoding to City_Area (Ordinal scale)
    # Instead of using the city_area names, we classified the 10 percentiles of the price histogram.
    # For each city, we aggregated the average prices of its city_areas and classified the city_areas using the percentiles.
    # Later, we converted the percentiles into weights on a scale of 1-10.
    # 10==Expensive city_area, 1==Cheap city_area.
    # The rank column is per city state ratio, therefore most likely that all houses in אוסישקין נהריה will be ranked as 1 and all house in שכונת הגולף קיסריה will be ranked as 10.
    
    percentiles = data['price'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    percentiles_dict = percentiles.to_dict()
    
    data['City']= data['City'].str.strip()
    grouped_data = data.groupby(['City', 'city_area'])['price'].mean()
    grouped_data = grouped_data.to_frame().reset_index()
    grouped_data = grouped_data.sort_values(['City', 'price'])
    def classify_rank(price):
        for percentile, threshold in percentiles_dict.items():
            if price <= threshold:
                return int(percentile * 10)
    grouped_data['rank'] = grouped_data['price'].apply(classify_rank)
    data = pd.merge(data, grouped_data[['City', 'city_area', 'rank']], on=['City', 'city_area'], how='left')
    data = data.drop('city_area', axis=1)
    try :
        data = data.drop('Unnamed: 23',axis=1)
    except :
        pass
    
    return data
