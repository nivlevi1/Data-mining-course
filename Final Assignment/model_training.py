import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score, mean_squared_error
import requests
import pandas as pd
import os.path
import re
import time
import numpy as np 
import math
from datetime import datetime, timedelta
from madlan_data_prep import prepare_data
import pickle
#For disable the warning messages :
import warnings
warnings.filterwarnings('ignore')




data = pd.read_excel(r"C:\Users\Niv-Levi\Final Assigment\Dataset_for_test.xlsx")
data = pd.DataFrame(data)
data = prepare_data(data)



# NOTE
'''
We used applied Label encoding and One-hot encoding on the prepare_data (We wanted to calculate it before the Data split!)
Dropped columns on "prepare_data":
publishedDays Too much Nans
number_in_street  Could make mistake because the model will think these numbers have real value.
Street  #We cant apply one hot encoding to a lot to streets & multicollinearity with the street_area.
'''


# Define the data preparation function
def prepare_data_after_split(X):
    data = X.copy()
    
    # Filling Missing Area's by the Average Area of the floor:
    overall_mean = math.ceil(data['Area'].mean())
    data['Area'] = data.groupby('floor')['Area'].transform(lambda x: x.fillna(x.mean())if not x.isnull().all() else overall_mean)

    # Filling missing Room_number's by the Average room numbers of the floor:
    data['room_number'] = data.groupby('floor')['room_number'].transform(lambda x: x.fillna(math.ceil(x.mean() * 2) / 2))

    # Filling all total_floors Nan's, Private houses(Total_floors) == 0, else get the mean total floor of each floor.
    data.loc[(data['floor'] == 0) & (data['total_floors'].isna()), 'total_floors'] = 0
    
    # Filling missing total_floors missing values in floor average total_floors 
    overall_mean = math.ceil(data['total_floors'].mean())
    data['total_floors'] = data.groupby('floor')['total_floors'].transform(lambda x: x.fillna(math.ceil(x.mean()) if not x.isnull().all() else overall_mean))

    # Num of images
    data['num_of_images'].fillna(0, inplace=True)
     
    return data

# Split the data and preprocess using the pipeline
X = data.drop("price", axis=1)
y = data.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O']
cat_cols = [col for col in X_train.columns if (X_train[col].dtypes=='O')]

#pipeline:
numerical_pipeline = Pipeline([
    ('preparation_after_split', FunctionTransformer(prepare_data_after_split)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first'))
])

column_transformer = ColumnTransformer([
    ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols)
], remainder='drop')

pipeline = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNet(alpha=0.1, l1_ratio=0.9))
])

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=7, scoring='neg_mean_squared_error') #7 for the "Dataset_for_test", originally in the application we used cv=10

# Convert negative scores to positive
rmse_scores = np.sqrt(-cv_scores)

# Make predictions on the test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Calculate the average RMSE score
average_rmse = np.mean(rmse_scores)

# Calculate R-squared score
r2_score_value = r2_score(y_test, y_pred)

print("Cross-Validation RMSE scores:", rmse_scores)
print("Average RMSE score:", average_rmse)
print("R-squared score:", r2_score_value)


# # Save the trained model as a PKL file
# with open('trained_model.pkl', 'wb') as file:
#     pickle.dump(pipeline, file)
