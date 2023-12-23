import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost as xgb
import argparse
import lightgbm as lgb
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd 
import random
import pickle
from mlforecast import MLForecast
import wandb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import plotly.express as px
import logging
import sys
import holidays
import os
import warnings
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

tr_holidays = holidays.TR(list(range(2014, 2019)))
holiday_dates = []
for holiday_date, holiday_name in tr_holidays.items():
    holiday_dates.append(pd.Timestamp(holiday_date).date())

random.seed(42)
np.random.seed(42)
data_path = 'data/'
log_path = "logs/"
result_path = "results/"

 # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

def preprocess_wind_plant_data(df, feature_pipeline):

    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True) 
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    date_range = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='H')
    n_missing = len(date_range) - len(df)
    
    if n_missing != 0:
        df.set_index('datetime', inplace=True)
        print(f">>> {n_missing} records are missing, which will be filled with interpolation")
        new_df = pd.DataFrame(np.nan, index = date_range, columns = df.columns)
        new_df.index.name = 'datetime'
        new_df.loc[df.index] = df.values
        df = new_df.interpolate().reset_index()
    
    df['cos_windDir10m'] = np.cos(2 * np.pi * df['windDir10m'] / 360)
    df['cos_windDir100m'] = np.cos(2 * np.pi * df['windDir100m'] / 360)
    df.drop(['windDir10m', 'windDir100m'], axis=1, inplace=True)
    
    df['year'] = df['datetime'].dt.year
    df['cos_month'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    df['cos_day'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)
    df['cos_hour'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

    #df.set_index('datetime', inplace=True)

    df.rename(columns={"datetime": "ds", "prod": "y"}, inplace=True)
    df['unique_id'] = 0
    
    processed_df = feature_pipeline.preprocess(df)

    processed_df.set_index('ds', inplace=True)
    
    return processed_df

def preprocess_electricty_demand_data(df, feature_pipeline):

    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True) 
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    date_range = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='H')
    n_missing = len(date_range) - len(df)
    
    if n_missing != 0:
        df.set_index('datetime', inplace=True)
        print(f">>> {n_missing} records are missing, which will be filled with interpolation")
        new_df = pd.DataFrame(np.nan, index = date_range, columns = df.columns)
        new_df.index.name = 'datetime'
        new_df.loc[df.index] = df.values
        df = new_df.interpolate().reset_index()

    df['year'] = df['datetime'].dt.year.values
    df['cos_month'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    df['cos_week'] = np.cos(2 * np.pi * df['datetime'].dt.isocalendar().week.astype(float) / 52)
    df['cos_weekday'] = np.cos(2 * np.pi * df['datetime'].dt.weekday.astype(float) / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)
    df['cos_hour'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['is_holiday'] = df['datetime'].dt.date.isin(holiday_dates).astype(float)
    df['weekends_change_july_2017'] = ((df['datetime'] > pd.Timestamp(2017,7,1) ) & (df['datetime'].dt.day_name().isin(['Saturday', 'Sunday']))).astype(float)
   
    #df.set_index('datetime', inplace=True)
    df.rename(columns={"datetime": "ds", 'demand': "y"}, inplace=True)
    df['unique_id'] = 0
    
    processed_df = feature_pipeline.preprocess(df)

    processed_df.set_index('ds', inplace=True)
    
    return processed_df

def cos_month(ds):
    return np.cos(2 * np.pi * ds.month / 12)

def cos_week(ds):
    return np.cos(2 * np.pi * ds.isocalendar().week.astype(float) / 52)

def cos_weekday(ds):
    return np.cos(2 * np.pi * ds.weekday / 7)

def cos_day(ds):
    return np.cos(2 * np.pi * ds.dayofyear / 365)

def cos_hour(ds):
    return np.cos(2 * np.pi * ds.hour / 24)


def suggest_lightgbm_params(trial):

    param = {
    "objective": "regression",
    "metric": "squarederror",
    "verbosity": -1,
    #"boosting_type": "gbdt",
    "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
    "max_depth" : trial.suggest_int("max_depth", 10, 40, step=2),
    #"n_estimators": trial.suggest_int('n_estimators', 100, 200 , 25),
    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
    "bagging_freq": trial.suggest_int("bagging_freq", 1, 20),
    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
}

    # if param["booster"] in ["gbtree", "goss"]:
    #     # maximum depth of the tree, signifies complexity of the tree.
    #     param["max_depth"] = trial.suggest_int("max_depth", 10, 40, step=2)

    return param

def suggest_xgboost_params(trial):

    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        # use exact for small dataset.
        "tree_method": "approx",
        "n_estimators": trial.suggest_int('n_estimators', 50, 200 , 50),
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        #"n_jobs" : 1,
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 10, 22, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 20)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)


    return param

def data_provider(dt_name = 'electricity_demand', n_lags = 168, testing=True):
    # dt_names = ['electricity_demand', 'wind_plant']
    # dt_name = dt_names[1]
    # n_lags = seq_len = 1
    # cv = 5
    # horizon = 1

    seq_len = n_lags 

    lag_cols = ['lag'+str(i) for i in range(1, seq_len+1)] 
    feature_pipeline = MLForecast(models=[],
        freq='h',
        lags = list(range(1, seq_len + 1)),# , #[seq_len],
        # lag_transforms={
        # 1: [
        #     RollingMean(window_size=7*24),
        # ]},
        #target_transforms=[Differences([7*24])],
        #date_features=['year', cos_month, cos_week, cos_weekday, cos_day, cos_hour],
        #num_threads=8,
        )

    if dt_name == 'wind_plant':
        target = 'prod' 
        preprocess_func = preprocess_wind_plant_data
    else:
        target = 'demand' 
        preprocess_func = preprocess_electricty_demand_data

    train_df = pd.read_csv(data_path + f'{dt_name}/train.csv', index_col=0)

    if testing:
        test_df = train_df.iloc[-train_df.shape[0] // 10:].copy()
        train_df = train_df.drop(test_df.index).reset_index(drop=True)
    else:
        test_df = pd.read_csv(data_path + f'{dt_name}/test.csv', index_col=0)

    y_test = test_df[target].copy()
    test_df[target] = -9999999
    all_df = pd.concat((train_df, test_df)).reset_index(drop=True)
  
    processed_data_lst = []

    for i, df in enumerate([train_df, all_df]):  

        processed_df = preprocess_func(df, feature_pipeline)

        if i > 0:
            processed_df = processed_df.iloc[-test_df.shape[0]:].copy()
            processed_df.drop('y', axis=1, inplace=True)
        
            processed_df[lag_cols] = processed_df[lag_cols].replace(-9999999, np.nan)

        processed_data_lst.append(processed_df)

    X_train, X_test = processed_data_lst
    y_train = X_train['y'].copy()
    
    if testing:
        X_val = X_train.iloc[-X_train.shape[0]//10:].copy()

        X_train.drop(X_val.index, inplace=True)
        y_train = X_train['y'].copy()
        X_train.drop('y', axis=1, inplace=True)

        y_val = X_val['y'].copy()
        X_val.drop('y', axis=1, inplace=True)
    else:
        X_val, y_val = None, None

    return X_train, y_train, X_val, y_val, X_test, y_test


# ts_split = TimeSeriesSplit(n_splits=cv, test_size=X_train.shape[0]//10)

# idx_lst = []

# for i, (train_index, val_index) in enumerate(ts_split.split(X_train)):
#     if i < cv - 1:
#         t_idx = train_index.tolist() + list(range(val_index[-1] + n_lags, X_train.shape[0]))
#     else:
#         t_idx = train_index.tolist() 
    
#     idx_lst.append((t_idx, val_index))

    
def objective(trial, model_type, datasets):
    
    X_train, y_train, X_val, y_val, X_test, y_test = datasets

    if model_type == 'xgboost':
        params = suggest_xgboost_params(trial)
                
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params, dtrain)
        pred_val = model.predict(dvalid)

    else:
        params = suggest_lightgbm_params(trial)

        dtrain = lgb.Dataset(X_train, label=y_train)
        #dvalid = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(params, dtrain)

        pred_val = model.predict(X_val)

    mae_val = mean_absolute_error(y_val, pred_val)

    return -mae_val

def test_prediction(model, seq_len, X_test):

    test_index = X_test.index.copy()
    X_test = X_test.reset_index(drop=True)

    pred_test = np.empty(len(X_test))
    pred_test[:] = np.nan

    
    lag_cols = ['lag'+str(i) for i in range(1, seq_len+1)] 
    for i in range(X_test.shape[0]):
        record = X_test.loc[i].values.reshape(1,-1)
        if np.isnan(record).sum():
            raise Exception
        pred_test[i] = model.predict(record)
        
        if i == X_test.shape[0] - 1:
            continue
        
        if i > seq_len - 1:
            lag_cols_to_fill = lag_cols
        else:
            lag_cols_to_fill = ['lag' + str(lag+1) for lag in range(i + 1)] 

        b = max(0, i-seq_len+1)
        X_test.loc[i+1, lag_cols_to_fill] = pred_test[b:i+1]
 
    X_test.index = test_index      

    return X_test, pred_test

def gradboost_tuning_main(dt_name, target, model_type, n_lags=168, n_trials=10, timeout=100, n_jobs=-1, testing=True):
    
    model_save_file_name = f"{dt_name}_final_{model_type}_lag{n_lags}.json"
    
    datasets = data_provider(dt_name, n_lags, testing)
    X_train, y_train, X_val, y_val, X_test, y_test = datasets
    
    # all_X_train = pd.concat((X_train, X_val))
    # all_y_train = pd.concat((y_train, y_val))
    
    if not os.path.exists(log_path + model_save_file_name):

        # wandb_kwargs = {"project": f"optuna-{model_type}-{dt_name}_forecasting"}
        # wandbc = WeightsAndBiasesCallback(metric_name="mae_val", wandb_kwargs=wandb_kwargs)
        print(">>> Starting optuna study....")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
        study.optimize(lambda trial: objective(trial, model_type, datasets), n_trials=n_trials, timeout=timeout,  n_jobs=n_jobs) # callbacks=[wandbc],
        best_trial = study.best_trial

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        print("  Value: ", best_trial.value)
        print("  Params: ")
        print(best_trial.params)

        if model_type == 'xgboost':
            model = XGBRegressor(**best_trial.params)  
            model.fit(X_train, y_train)
            model.save_model(log_path + model_save_file_name)

        else:
            model = LGBMRegressor(**best_trial.params)
            model.fit(X_train, y_train)
            model.booster_.save_model(log_path + model_save_file_name)

        with open(log_path + f'{dt_name}_best_{model_type}_lag{n_lags}_trial.pkl', 'wb') as f:
            pickle.dump(best_trial, f)

        with open(log_path + f'{dt_name}_{model_type}_lag{n_lags}_study.pkl', 'wb') as f:
            pickle.dump(study, f)

    else:

        with open(log_path + f'{dt_name}_best_{model_type}_lag{n_lags}_trial.pkl', 'rb') as f:
            best_trial = pickle.load(f)

        if model_type == 'xgboost':
            model = XGBRegressor()  
            model.load_model(log_path + model_save_file_name)
        else:
            model = LGBMRegressor()
            model = lgb.Booster(model_file = log_path + model_save_file_name)


    X_test_filled_lags, pred_test = test_prediction(model, n_lags, X_test)
    X_test[target] = y_test.values
    X_test['pred'] = pred_test

    if not testing:
        future_df = pd.read_csv(data_path + f'{dt_name}/test.csv', index_col=0)

        future_df[target] = pred_test

        future_df.to_csv(result_path + f"pred_{dt_name}_final_{model_type}_lag{n_lags}.csv")
        print('>>> Saved:', result_path + f"pred_{dt_name}_final_{model_type}_lag{n_lags}.csv")

        fig = px.line(future_df[target])
        fig.update_layout(title = f'Forecasting for Future {dt_name} Dataset (test.csv) with {model_type}', title_x = 0.5)
        fig.show()
    else:
        print("Test MAE:", np.mean(np.abs(y_test - np.array(pred_test))))

        fig = px.line(X_test[[target, 'pred']])
        fig.update_layout(title = f'Test Forecasting for {dt_name} Dataset with {model_type}', title_x = 0.5)
        fig.show()

        # pred_val = model.predict(X_val)

        # X_val[target] = y_val
        # X_val['pred'] = pred_val

        # print("Val. MAE:", np.mean(np.abs(y_val - np.array(pred_val))))

        # fig = px.line(X_val[[target, 'pred']])
        # fig.show()

    return 

if __name__ == "__main__":

    model_type = 'lightgbm' # or xgboost

    parser = argparse.ArgumentParser()
    parser.add_argument('--dt_name', type=str, default='electricity_demand')
    parser.add_argument('--model_type', type=str, default='lightgbm')
    parser.add_argument('--n_lags', type=int)
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--testing', type=int, default=1) 
    # testing = 1: score the best model over the splitted testing data from train.csv 
    # testing = 0: use the best model to predict test.csv

    args = parser.parse_args()
    
    dt_name = args.dt_name
    target = 'demand' if dt_name == 'electricity_demand' else 'prod'
    model_type = args.model_type
    n_lags = args.n_lags if not args.n_lags is None else {'wind_plant':1, 'electricity_demand':24*7}[dt_name]
    n_trials = args.n_trials
    timeout = args.timeout
    n_jobs = args.n_jobs
    testing = args.testing    

    gradboost_tuning_main(dt_name, target, model_type, n_lags, n_trials, timeout, n_jobs, testing)