
import pandas as pd
import numpy as np


from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn import model_selection

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

import joblib

import argparse
import config 
import models_dispatcher 

import itertools
import config

def feature_engineering(df):
    #lib

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

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))

    #some more-feature engineering:

    df["TotalGarageQual"] = df["GarageQual"] * df["GarageCond"]
    df["TotalExteriorQual"] = df["ExterQual"] * df["ExterCond"]
    

    #df.drop(["PoolQC"],axis=1,inplace=True)

    # box_cox

    numeric_feats = [feature for feature in df.columns if df[feature].dtype != "object" and feature not in ("Id", "kfold","SalePrice")]
    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    
    skewness = skewness[abs(skewness) > 0.75]
    
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], lam)


    #rare features 
    features = [feature for feature in df.columns if df[feature].dtype == 'O']   

    for feature in features:
        abc = df[feature].value_counts().to_dict()
        for key, value in abc.items():
            if value/len(df[feature]) < 0.01:
                df.loc[:,feature][df[feature]==key]="rare"

    return df


def run(fold,model):

    #loading
    df = pd.read_csv(config.training_data_with_folds)
    df_test = pd.read_csv(config.test_data_loc)

    #function run
    df = feature_engineering(df)
    df_test = feature_engineering(df_test)  

     #Some missing values still-were comming!! (please fix it in the above feature-eng(function) otherwise this also works fine)
    df_test.fillna(0,inplace=True) 

    df.SalePrice = np.log1p(df.SalePrice)

    #concat
    df = pd.concat([df,df_test],axis=0)
    
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

    numeric_features = [feature for feature in df_train.columns if feature not in ("Id", "kfold","SalePrice")]
    
    x_train = df_train[numeric_features].values
    x_valid = df_valid[numeric_features].values

    
    #regressor models

    reg = models_dispatcher.MODELS[model]

    reg.fit(x_train,df_train.SalePrice.values)
    valid_preds = reg.predict(x_valid)
    test_preds = reg.predict(df_test[numeric_features].values)
    
    
    #joblib.dump(features, os.path.join(config.models_location, f"{model}_{fold}_columns.bin"))

    # scoring
    
    rmse = np.sqrt(mean_squared_error(df_valid.SalePrice.values, valid_preds))
    mae = mean_absolute_error(df_valid.SalePrice.values, valid_preds)
    
    print(f"FOLD={fold}, MODEL = {model}, RMSE = {rmse}, MAE ={mae} ")
    return(test_preds)

preds_df = pd.DataFrame()

for fold in range(3):
    for keys,items in models_dispatcher.MODELS.items():    
        preds_df["fold"+str(fold)+keys] = run(fold,keys)

for cols in preds_df.columns:
    preds_df[cols] = np.expm1(preds_df[cols])

for e,col in enumerate(preds_df.columns):
    if e == 0:
        preds_df['SalePrice'] = preds_df[col]
    else:
        preds_df['SalePrice'] += preds_df[col]    


preds_df['SalePrice'] =  preds_df['SalePrice']/18

df_test = pd.read_csv(config.test_data_loc,usecols=["Id"])
preds_df["id"] = df_test.values.flatten()

abc  = preds_df[['id','SalePrice']]
abc.to_csv(f"/home/hazim/Desktop/Advanced-house-price-prediction/output/abc.csv", index=False)