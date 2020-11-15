from sklearn import linear_model
from sklearn import ensemble

MODELS = {
    "lr": linear_model.LogisticRegression(),
    "rf": ensemble.RandomForestRegressor(n_jobs=-1)
}