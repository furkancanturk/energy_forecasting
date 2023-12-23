# Energy Forecasting Project

This project involves two forecasting problems. 
* Hourly energy demand forecasting
* Hourly wind power plant production forecasting 

[iTransformer](https://github.com/thuml/iTransformer), NeuralProphet, XGBoost and LightGBM models are implemented for next 1-hour and 24-hour forecasting. 

The objective is providing forecasting for test.csv datasets which comprise only data of exogenous variables for a long horizon and do not include the time series of target variables.

Tested with Python 3.9.13 environment installed with pip.

> [!NOTE]
> The datasets train.csv and test.csv in data/electricity_demand/ and data/wind_plant/ folders are private; therefore, they haven't been uploaded.

