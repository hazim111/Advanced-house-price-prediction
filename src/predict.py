import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn import preprocessing
import numpy as np

import os
import joblib

import argparse
import config 
import models_dispatcher 

import itertools
import config



def feature_engineering(df):

    temporal_features = [feature for feature in df.columns if 'Yr' in feature or 'Year' in feature or 'Mo' in feature]
    numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O' and feature not in temporal_features and feature not in ("Id", "kfold","SalePrice")]
    categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O' and feature not in temporal_features]

    
    #feature-eng on temporal-dataset

    for feature in temporal_features:
        if feature == 'YrSold' or feature == 'MoSold':
            pass
        else:
            df[feature] = df['YrSold'] - df[feature]

    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str) 
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    
    
    #fill-na

    for feature in numeric_features:
        df[feature] = df.groupby("Neighborhood")[feature].transform(lambda x: x.fillna(x.median()))

    for feature in categorical_features:
        df[feature] = df[feature].fillna("Missing")

    for feature in temporal_features:
        if feature == 'YrSold' or feature == 'MoSold':
            df[feature] = df[feature].fillna("Missing")
        else:
            df[feature] = df[feature].fillna(0)

    #feature-generation

    df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['TotalLot'] = df['LotFrontage'] + df['LotArea']

    df['TotalBsmtFin'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    
    df['TotalBath'] = df['FullBath'] + df['HalfBath']

    df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch']

    #feature-selection (multi-correnality)

    #df.drop(['TotalBsmtFin','LotArea','TotalBsmtSF','GrLivArea','GarageYrBlt','GarageArea'],axis=1,inplace=True)

    return df

def predict(model):
    df = pd.read_csv(config.test_data_loc)
    test_idx = df["Id"].values
    predictions = None

    for fold in range(5):
        df = pd.read_csv(config.test_data_loc)
        df = feature_engineering(df) 
        
        encoders = joblib.load(os.path.join(config.models_location, f"{model}_{fold}_label_encoder.bin"))
        cols = joblib.load(os.path.join(config.models_location, f"{model}_{fold}_columns.bin"))
        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
        
        reg = joblib.load(os.path.join(config.models_location, f"{model}_{fold}.bin"))
        
        df = df[cols]
        preds = reg.predict(df)

        if fold == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    submission = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "SalePrice"])
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"{model}.csv", index=False)
    return submission
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    predict(model=args.model)
