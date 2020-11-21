from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor


#elasticnet
elasticnet_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
elasticnet_l1ratios = [0.8, 0.85, 0.9, 0.95, 1]
#lasso
lasso_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
#ridge
ridge_alphas = [13.5, 14, 14.5, 15, 15.5]


MODELS = { 
    "elasticnet" : make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=elasticnet_alphas, l1_ratio=elasticnet_l1ratios)),
     "lasso" : make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=lasso_alphas, random_state=42)),
     "ridge" : make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas)),
     "gradb" : GradientBoostingRegressor(n_estimators=6000, learning_rate=0.01,
                                  max_depth=4, max_features='sqrt',
                                  min_samples_leaf=15, min_samples_split=10,
                                  loss='huber', random_state=42),

    "svr" : make_pipeline(RobustScaler(),
                    SVR(C=20, epsilon=0.008, gamma=0.0003)),

    "xgboost" : XGBRegressor(learning_rate=0.01, n_estimators=6000,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006, random_state=42)}

MODELS_stack = StackingCVRegressor(regressors=(MODELS['elasticnet'], MODELS['gradb'], MODELS['lasso'], 
                                          MODELS['ridge'], MODELS['svr'], MODELS['xgboost']),
                              meta_regressor=MODELS['xgboost'],
                              use_features_in_secondary=True)