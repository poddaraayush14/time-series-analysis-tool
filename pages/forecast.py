# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 21:09:53 2021

@author: podda
"""

import os
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor

scaler = StandardScaler()
sc1 = StandardScaler()

#st.set_page_config(layout="wide")

def app():
    typ = st.radio("Select the type of Time Series",("Univariate", "MultiVariate"))
    
    if typ == 'Univariate':
        if 'udata.csv' not in os.listdir('data'):
            st.markdown("Please upload Univariate data according to the template through the 'Upload Data' page!")
            
        else:
            ts = pd.read_csv('data/udata.csv')
            
            forecast_out = st.number_input('Enter the number of periods you want to forecast in future', min_value=1, max_value=15)
            
            #ts = ts.fillna(method='ffill').fillna(method='bfill') 
            tf = pd.DataFrame(sc1.fit_transform(ts), columns = ts.columns)
            def dp(df):
                dfu = df.copy()
                dfu['Prediction'] = dfu[['y']].shift(-forecast_out)
                X = dfu.drop(['Prediction'], axis = 1)
                X = X[:-forecast_out]
                y = dfu['Prediction']
                y = y[: -forecast_out]
                x_forecastu = dfu.drop(['Prediction'],axis = 1)[-forecast_out:]
                x_train1, x_test1, y_train1, y_test1 = X[: int(0.8*len(X))], X[int(0.8*len(X)) :], y[: int(0.8*len(X))], y[int(0.8*len(X)) :]
                return x_train1, x_test1, y_train1, y_test1, x_forecastu
            st.write('Select models to display their corresponding forecast:')
            col1,col2 = st.beta_columns(2)  
            sar = col1.checkbox('ARIMA/SARIMA')
            hwe = col1.checkbox('Holt Winters Exponential Smoothing')
            svreg = col2.checkbox('Simple Vector Regression (SVR)')
            en = col2.checkbox('Ensemble ML')
            
            if sar is True:
                mo = SARIMAX(tf[:-forecast_out],order=(2, 1, 2),seasonal_order=(0, 0, 0, 6))
                mo_fit = mo.fit()
                mo1 = ARIMA(tf[:-forecast_out], order=(1, 0, 0))
                mo1_fit = mo1.fit()
                cs1 = mo_fit.forecast(steps=forecast_out)
                cs2 = mo1_fit.forecast(steps=forecast_out)
                cf = (cs1 + cs2[0])/2
                mae1 = mean_absolute_error(ts[-forecast_out:], sc1.inverse_transform(cf))
                ms = SARIMAX(tf,order=(2, 1, 2),seasonal_order=(0, 0, 0, 6))
                ms_fit = ms.fit()
                ma = ARIMA(tf, order=(1, 0, 0))
                ma_fit = ma.fit()
                gp = pd.DataFrame()
                csd1 = ms_fit.forecast(steps=forecast_out)
                csd2 = ma_fit.forecast(steps=forecast_out)
                cfd = (csd1 + csd2[0])/2
                gp['Forecast'] = sc1.inverse_transform(cfd)
                st.subheader("Forecast using ARIMA/SARIMA")
                st.write(gp)                
                st.write("MAE : ", mae1)
                
            if hwe is True:
                hwmod = HWES(tf[:-forecast_out], seasonal_periods=30, trend='add', seasonal='add')
                hwfit = hwmod.fit()
                mae2 = mean_absolute_error(ts[-forecast_out:], sc1.inverse_transform(hwfit.forecast(steps=forecast_out)))
                mod = HWES(tf, seasonal_periods=30, trend='add', seasonal='add')
                fitted = mod.fit()
                gp = pd.DataFrame()
                gp['Forecast'] = sc1.inverse_transform(fitted.forecast(steps=forecast_out))
                st.subheader("Forecast using Holt Winters Exponential Smoothing")
                st.write(gp)
                st.write("MAE : ", mae2)
                
            if svreg is True:
                x_train, x_test, y_train, y_test, x_forecast = dp(tf)
                svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.01)
                svr_rbf.fit(x_train, y_train)           
                sv_prediction = svr_rbf.predict(x_forecast)
                gp = pd.DataFrame()
                gp['Forecast'] = sc1.inverse_transform(sv_prediction)
                y_psvr = sc1.inverse_transform(svr_rbf.predict(x_test))
                mae3 = mean_absolute_error(sc1.inverse_transform(y_test), y_psvr)
                pfig3 = plt.figure()
                plt.title("Fit of the Forecast - SVR")
                plt.xlabel("Timestamp")
                plt.ylabel("y")
                plt.plot(sc1.inverse_transform(y_test), label = 'Actual')
                plt.plot(y_psvr, label='Forecast', color='red')
                plt.legend()
                st.plotly_chart(pfig3)
                st.subheader("Forecast using SVR")
                st.write(gp)
                st.write("MAE : ", mae3)
                
            if en is True:
                estimators = [('rf', RandomForestRegressor(bootstrap=True,max_depth=5, max_features='sqrt',
                                                  min_samples_leaf=25,
                                                  min_samples_split=16,
                                                  n_estimators=153, n_jobs=-1,
                                                  random_state=7)),
                                 ('svr', SVR(kernel='rbf', C=0.1, gamma=10)),
                                 ('xgb', xgb.XGBRegressor(gamma=0.02, learning_rate= 0.05, max_depth=8, n_estimators= 100, objective='reg:squarederror'))
                                 ]
                x_train, x_test, y_train, y_test, x_forecast = dp(tf)
                stk = StackingRegressor(estimators=estimators,final_estimator=LinearRegression())
                stk.fit(x_train, y_train)           
                st_prediction = stk.predict(x_forecast)
                gp = pd.DataFrame()
                gp['Forecast'] = sc1.inverse_transform(st_prediction)
                y_pst = sc1.inverse_transform(stk.predict(x_test))
                mae4 = mean_absolute_error(sc1.inverse_transform(y_test), y_pst)
                pfig4 = plt.figure()
                plt.title("Fit of the Forecast - Stacking Ensemble")
                plt.xlabel("Timestamp")
                plt.ylabel("y")
                plt.plot(sc1.inverse_transform(y_test), label = 'Actual')
                plt.plot(y_pst, label='Forecast', color='red')
                plt.legend()
                st.plotly_chart(pfig4)
                st.subheader("Forecast using Ensemble Stacking")
                st.write(gp)
                st.write("MAE : ", mae4)
    
    else:
        if 'mdata.csv' not in os.listdir('data'):
            st.markdown("Please upload data (either univariate or multivariate according to template through 'Upload Data' page!")
        
        else:
            
            data = pd.read_csv('data/mdata.csv')
            forecast_out = st.number_input('Enter the number of periods you want to forecast in future', min_value=1, max_value=15)
            
            #data = data.fillna(method='ffill').fillna(method='bfill')
            df = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
            dfg = data[['y']]
            tst = sc1.fit_transform(dfg)
            train, test = df[0: int(0.8*len(df))], df[int(0.8*len(df)) :]
            test = test.reset_index()
            test.drop(['index'], axis=1, inplace=True)
                  
            def dp(df):
                df1 = df.copy()
                df1['Prediction'] = df1[['y']].shift(-forecast_out)
                X = df1.drop(['Prediction'], axis = 1)
                X = X[:-forecast_out]
                y = df1['Prediction']
                y = y[: -forecast_out]
                x_forecast = df1.drop(['Prediction'],axis = 1)[-forecast_out:]
                x_train, x_test, y_train, y_test = X[: int(0.8*len(X))], X[int(0.8*len(X)) :], y[: int(0.8*len(X))], y[int(0.8*len(X)) :]
                return x_train, x_test, y_train, y_test, x_forecast
            
            st.write('Select models to display their corresponding forecast:')
            col1,col2 = st.beta_columns(2)
            var = col1.checkbox('Vector Auto Regression (VAR)')
            varm = col1.checkbox('Vector Auto Regression Moving Average (VARMA)')
            svreg = col2.checkbox('Simple Vector Regression (SVR)')
            xg = col2.checkbox('Xtreme Gradient Boosting (XGBoost)')
            
            if var is True:
                def predict(coef, history):
                	yhat = coef['y'][0]
                	for i in range(1, len(coef)):
                		yhat += coef['y'][i] * history[-i]
                	return yhat              
                model = VAR(train)
                results = model.fit(2) 
                coef = results.params
                history = [train['y'][i+1] for i in range(len(train)-1)]
                predictions = list()
                for t in range(len(test)):
                	yhat = predict(coef, history)
                	obs = test['y'][t]
                	predictions.append(yhat)
                	history.append(obs)
                pfig = plt.figure()
                plt.title("Fit of the Forecast - VAR")
                plt.xlabel("Timestamp")
                plt.ylabel("y")
                plt.plot(sc1.inverse_transform(test['y']), label='Actual')
                plt.plot(sc1.inverse_transform(predictions), label='Forecasted',color='red')
                plt.legend()
                st.plotly_chart(pfig)
                
                mae1 = mean_absolute_error(sc1.inverse_transform(test['y']), sc1.inverse_transform(predictions))
                fmodel = VAR(df)
                fvar = fmodel.fit(2)
                a = fvar.forecast(df.values[-fvar.k_ar:],steps=forecast_out)
                pred = []
                for i in range(forecast_out):
                  pred.append(a[i][0]) 
                gp = pd.DataFrame()
                gp['Predicted'] = sc1.inverse_transform(pred)
                st.subheader("Forecast using Vector Auto Regression")
                st.write(gp)
                st.write("MAE : ", mae1)
            
            if varm is True:
                def predictvm(coef, history):
                    yhat = coef[0]
                    for i in range(1, len(coef)):
                    	yhat += coef[i] * history[-i]
                    return yhat
                
                modelvm = VARMAX(train, order = (2,2))
                model_fit = modelvm.fit()
                coef1 = model_fit.params
                c1 = coef1[0:1]
                c1 = c1.append(coef1[4:12])
                vhistory = [train['y'][i+1] for i in range(len(train)-1)]
                vpredictions = list()
                for t in range(len(test)):
                	yhat = predictvm(c1, vhistory)
                	obs = test['y'][t]
                	vpredictions.append(yhat)
                	vhistory.append(obs)
                pfig2 = plt.figure()
                plt.title("Fit of the Forecast - VARMAX")
                plt.xlabel("Timestamp")
                plt.ylabel("y")
                plt.plot(sc1.inverse_transform(test['y']), label = 'Actual')
                plt.plot(sc1.inverse_transform(vpredictions), label='Forecast', color='red')
                plt.legend()
                st.plotly_chart(pfig2)
                mae2 = mean_absolute_error(sc1.inverse_transform(test['y']), sc1.inverse_transform(vpredictions))
                fmodelv = VARMAX(df, order = (2,2))
                fvarm = fmodelv.fit()
                am = fvarm.forecast(steps=forecast_out)
                gp = pd.DataFrame()
                gp['Forecast'] = sc1.inverse_transform(am['y'].values)
                st.subheader("Forecast using VARMAX")
                st.write(gp)
                st.write("MAE : ", mae2)
                
            if svreg is True:
                x_train, x_test, y_train, y_test, x_forecast = dp(df)
                svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.01)
                svr_rbf.fit(x_train, y_train)           
                sv_prediction = svr_rbf.predict(x_forecast)
                gp = pd.DataFrame()
                gp['Forecast'] = sc1.inverse_transform(sv_prediction)
                y_psvr = sc1.inverse_transform(svr_rbf.predict(x_test))
                mae3 = mean_absolute_error(sc1.inverse_transform(y_test), y_psvr)
                pfig3 = plt.figure()
                plt.title("Fit of the Forecast - SVR")
                plt.xlabel("Timestamp")
                plt.ylabel("y")
                plt.plot(sc1.inverse_transform(y_test), label = 'Actual')
                plt.plot(y_psvr, label='Forecast', color='red')
                plt.legend()
                st.plotly_chart(pfig3)
                st.subheader("Forecast using SVR")
                st.write(gp)
                st.write("MAE : ", mae3)
            if xg is True:
                x_train, x_test, y_train, y_test, x_forecast = dp(df)
                modelxg = xgb.XGBRegressor(gamma=0.01, learning_rate= 0.01, max_depth=8,
                                           n_estimators= 300, objective='reg:squarederror')
                modelxg.fit(x_train,y_train)           
                xg_prediction = modelxg.predict(x_forecast)
                gp = pd.DataFrame()
                gp['Forecast'] = sc1.inverse_transform(xg_prediction)
                y_pxg = sc1.inverse_transform(modelxg.predict(x_test))
                mae4 = mean_absolute_error(sc1.inverse_transform(y_test), y_pxg)
                pfig4 = plt.figure()
                plt.title("Fit of the Forecast - XGBoost")
                plt.xlabel("Timestamp")
                plt.ylabel("y")
                plt.plot(sc1.inverse_transform(y_test), label = 'Actual')
                plt.plot(y_pxg, label='Forecast', color='red')
                plt.legend()
                st.plotly_chart(pfig4)
                st.subheader("Forecast using XGBoost")
                st.write(gp)
                st.write("MAE : ", mae4)
