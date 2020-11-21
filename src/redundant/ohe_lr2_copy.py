import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
import numpy as np

from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

import os
import joblib

import argparse
import config 
import models_dispatcher 

import itertools
import config



def feature_engineering(df):

    temporal_features = [feature for feature in df.columns if 'Yr' in feature or 'Year' in feature or 'Mo' in feature]
    numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O' and feature not in temporal_features and feature not in ("Id", "kfold")]
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

    #df.drop(["PoolQC"],axis=1,inplace=True)

    return df


def run(fold,model):

    #loading
    df = pd.read_csv(config.training_data_with_folds)
    df_test = pd.read_csv(config.test_data_loc)

    df_test.loc[:,'SalePrice'] = -1
    df_test.loc[:,'kfold'] = 72

    df = pd.concat([df,df_test],axis=0)
    #function run
    df = feature_engineering(df)  


    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))

    

    df["TotalGarageQual"] = df["GarageQual"] * df["GarageCond"]
    df["TotalExteriorQual"] = df["ExterQual"] * df["ExterCond"]



    # normailise numeric_features:

    numeric_feats = [feature for feature in df.columns if df[feature].dtype != "object" and feature not in ("Id", "kfold","SalePrice")]
    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    
    skewness = skewness[abs(skewness) > 0.75]
    
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], lam)



    df.SalePrice = np.log1p(df.SalePrice)



    features = [feature for feature in df.columns if df[feature].dtype == 'O']   

    for feature in features:
        abc = df[feature].value_counts().to_dict()
        for key, value in abc.items():
            if value/len(df[feature]) < 0.01:
                df.loc[:,feature][df[feature]==key]="rare"

    df = pd.get_dummies(df)
    
    #Feature_selection

    for feature in df.columns:
        all_value_counts = df[feature].value_counts()
        zero_value_counts = all_value_counts.iloc[0]
        if zero_value_counts / len(df) > 0.99:
            df.drop(feature,axis=1,inplace=True)

    #split back
    
    df_test = df.loc[df["Id"].between(1461,2919)]

    df =  df.loc[df["Id"].between(1,1460)]
    

    
    #train-tests-split
    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["SalePrice"],axis=1)
    x_valid = df_valid.drop(["SalePrice"],axis=1)

    numeric_features = [feature for feature in x_train.columns if feature not in ("Id", "kfold")]
    

    reg = models_dispatcher.MODELS[model]
    reg.fit(x_train[numeric_features],df_train.SalePrice.values)
    valid_preds = reg.predict(x_valid[numeric_features])
    
    #joblib.dump(features, os.path.join(config.models_location, f"{model}_{fold}_columns.bin"))

    # scoring
    
    rmse = np.sqrt(mean_squared_error(df_valid.SalePrice.values, valid_preds))
    mae = mean_absolute_error(df_valid.SalePrice.values, valid_preds)
    
    print(f"FOLD={fold}, RMSE = {rmse}, MAE ={mae} ")

    joblib.dump(reg, os.path.join(config.models_location,f"{model}_{fold}.bin"))
    #joblib.dump(var_thresh, os.path.join(config.models_location,f"{model}_{fold}_var_thresh.bin"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold",type=int)


    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    run(fold= args.fold, model=args.model)

