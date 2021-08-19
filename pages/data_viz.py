# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:53:43 2021

@author: podda
"""
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import numpy as np


def app():
    typ = st.radio("Select the type of Time Series",("Univariate", "MultiVariate"))
    if typ == 'Univariate':
        if 'udata.csv' not in os.listdir('data'):
            st.markdown("Please upload Univariate data according to the template through the 'Upload Data' page!")
        else:
            ts = pd.read_csv('data/udata.csv')
            
            st.write('Select plots to display:')
            col1,col2,col3 = st.beta_columns(3)
            desc = col1.checkbox('Dataset Description')
            tsr = col2.checkbox('Trend/ Seasonality and Residual')
            acp = col3.checkbox('ACF/ PACF Plot')

            if desc is True:
                st.subheader("Plot of the target variable")
                pfig1 = plt.figure()
                plt.title("Dataset Target Variable Plot")
                plt.xlabel("Timestamp")
                plt.ylabel("y")
                plt.plot(ts['y'], label = 'y-value', color='green')
                plt.legend()
                st.plotly_chart(pfig1, use_container_width=True)
                st.write(ts.describe())
            
            if tsr is True:
                st.subheader("Decomposition of Time Series target variable into Trend/Seasonality and Residue")
                pfig2 = plt.figure()
                plt.title("Trend Seasonality and Residual Decomposition Plot")
                decomposed = sm.tsa.seasonal_decompose(ts['y'].values, freq=100)
                pfig2 = decomposed
                st.write(pfig2.plot())
                
            if acp is True:
                st.subheader("Autocorrelation Function and Partial Autocorrelation Function ")
                lag_acf = acf(ts['y'], nlags=200)
                lag_pacf = pacf(ts['y'], nlags=20, method='ols')
                
                pfig3 = plt.figure()
                plt.plot(lag_acf)
                plt.axhline(y=0,linestyle='--',color='gray')
                plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
                plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
                plt.title('Autocorrelation Function')
                st.write(pfig3)
                
                pfig4 = plt.figure()
                plt.plot(lag_pacf)
                plt.axhline(y=0,linestyle='--',color='gray')
                plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
                plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
                plt.title('Partial Autocorrelation Function')
                st.write(pfig4)
    
    else:
        if 'mdata.csv' not in os.listdir('data'):
            st.markdown("Please upload Multivariate data according to the template through the 'Upload Data' page!")
        else:
            df = pd.read_csv('data/mdata.csv')
            
            st.write('Select plots to display:')
            col1,col2,col3 = st.beta_columns(3)
            desc = col1.checkbox('Dataset Description')
            tsr = col2.checkbox('Trend/ Seasonality and Residual')
            acp = col3.checkbox('ACF/ PACF Plot')
            
            if desc is True:
                st.subheader("Plots for features and target variable")
                fig1 = plt.figure()
                plt.title("Dataset Target Variable Plot")
                plt.xlabel("Timestamp")
                plt.ylabel("y")
                plt.plot(df['y'], label = 'y-value', color = 'red')
                plt.legend()
                st.plotly_chart(fig1)
                st.subheader("Plots for the independent variables")
                for i in range((len(df.columns)-1)):
                    figa = plt.figure()
                    plt.xlabel("Timestamp")
                    plt.ylabel(df.columns[i+1])
                    plt.plot(df[df.columns[i+1]], color = 'green')
                    st.plotly_chart(figa)
                st.subheader("Description of the Time Series")
                st.write(df.describe())
                
            if tsr is True:
                st.subheader("Decomposition of Time Series target variable into Trend/Seasonality and Residue")
                fig2 = plt.figure()
                plt.title("Trend Seasonality and Residual Decomposition Plot")
                decomposed = sm.tsa.seasonal_decompose(df['y'].values, freq=100)
                fig2 = decomposed
                st.write(fig2.plot())
            
            if acp is True:
                st.subheader("Autocorrelation Function and Partial Autocorrelation Function ")
                lag_acf = acf(df['y'], nlags=200)
                lag_pacf = pacf(df['y'], nlags=20, method='ols')
                
                fig3 = plt.figure()
                plt.plot(lag_acf)
                plt.axhline(y=0,linestyle='--',color='gray')
                plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='gray')
                plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='gray')
                plt.title('Autocorrelation Function')
                st.write(fig3)
                
                fig4 = plt.figure()
                plt.plot(lag_pacf)
                plt.axhline(y=0,linestyle='--',color='gray')
                plt.axhline(y=-1.96/np.sqrt(len(df)),linestyle='--',color='gray')
                plt.axhline(y=1.96/np.sqrt(len(df)),linestyle='--',color='gray')
                plt.title('Partial Autocorrelation Function')
                st.write(fig4)       