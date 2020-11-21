import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import VarianceThreshold

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

    df.drop(["PoolQC"],axis=1,inplace=True)

    return df

def predict(model):
    df = pd.read_csv(config.test_data_loc)
    test_idx = df["Id"].values
    predictions0 = None
    predictions1 = None
    predictions2 = None
    predictions3 = None
    predictions4 = None
    
    for fold in range(5):
        df = pd.read_csv(config.training_data_with_folds)
        df_test = pd.read_csv(config.test_data_loc)

        df_test.loc[:,'SalePrice'] = -1
        df_test.loc[:,'kfold'] = 72

        df = pd.concat([df,df_test],axis=0)
        #function run
        df = feature_engineering(df)  

        df_test = df.loc[df["Id"].between(1461,2919)]

        df =  df.loc[df["Id"].between(1,1460)]


        # normailise numeric_features:

        numeric_features = [feature for feature in df_test.columns if df_test[feature].dtype != 'O' and feature not in ("Id", "kfold")]
        
        scaler = StandardScaler()

        scaler.fit(df_test[numeric_features])

        df_test[numeric_features] = scaler.transform(df_test[numeric_features])

        features = [feature for feature in df.columns if df[feature].dtype == 'O']   

        for feature in features:
            abc = df[feature].value_counts().to_dict()
            for key, value in abc.items():
                if value/len(df[feature]) < 0.01:
                    df.loc[:,feature][df[feature]==key]="rare"
        
        df = pd.concat([df,df_test],axis=0)

        df = pd.get_dummies(df)
        
        #Feature_selection

        for feature in df.columns:
            all_value_counts = df[feature].value_counts()
            zero_value_counts = all_value_counts.iloc[0]
            if zero_value_counts / len(df) > 0.99:
                df.drop(feature,axis=1,inplace=True)

        #split back
        
        df = df.loc[df["Id"].between(1461,2919)]

        numeric_features = [feature for feature in df.columns if feature not in ("Id", "kfold","SalePrice")]
    
        reg = joblib.load(os.path.join(config.models_location, f"{model}_{fold}.bin"))
            
        preds = reg.predict(df[numeric_features])

        if fold == 0:
            predictions0 = preds
        elif fold == 1:
            predictions1 = preds
        elif fold == 2:
            predictions2 = preds
        elif fold == 3:
            predictions3 = preds
        elif fold == 4:
            predictions4 = preds

    predictions0 = np.exp(predictions0)
    predictions1 = np.exp(predictions1)
    predictions2 = np.exp(predictions2)
    predictions3 = np.exp(predictions3)
    predictions4 = np.exp(predictions4)


    predictions = (predictions0+predictions1+predictions2+predictions3+predictions4)/5

    submission = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "SalePrice"])
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    #submission.loc[:, "SalePrice"] = np.exp(submission.loc[:, "SalePrice"])
    
    submission.to_csv(f"{model}.csv", index=False)
    return submission
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    predict(model=args.model)
