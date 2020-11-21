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

    df.drop(["PoolQC"],axis=1,inplace=True)

    return df


def run(fold,model):

    #loading
    df = pd.read_csv(config.training_data_with_folds)
    df_test = pd.read_csv(config.test_data_loc)

    #function run
    df = feature_engineering(df)  

    features = [feature for feature in df.columns if df[feature].dtype == 'O']

    for col in features:
        df_test[col] = df_test[col].astype(str).fillna("Missing")

    
    #
    label_encoders = {}
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col].values.tolist()+ df_test[col].values.tolist())
        df[col] = lbl.transform(df[col].values.tolist())
        label_encoders[col] = lbl 

    #train-tests-split
    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    features = [feature for feature in df.columns if feature not in ("SalePrice","Id","kfold")]

    x_train = df_train[features].values
    x_valid = df_valid[features].values


    # modelling

    reg = models_dispatcher.MODELS[model]

    reg.fit(x_train,df_train.SalePrice.values)

    valid_preds = reg.predict(x_valid)


    # scoring
    rmse = np.sqrt(mean_squared_error(df_valid.SalePrice.values, valid_preds))
    rmsle = np.sqrt(mean_squared_log_error(df_valid.SalePrice.values, valid_preds))
    mae = mean_absolute_error(df_valid.SalePrice.values, valid_preds)
    
    print(f"FOLD={fold}, RMSE = {rmse}, RMSLE ={rmsle}, MAE ={mae} ")

    joblib.dump(reg, os.path.join(config.models_location,f"{model}_{fold}.pkl"))
    joblib.dump(label_encoders, os.path.join(config.models_location,f"{model}_{fold}_label_encoder.pkl"))
    joblib.dump(features, os.path.join(config.models_location, f"{model}_{fold}_columns.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold",type=int)


    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    run(fold= args.fold, model=args.model)

