# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:53:02 2021

@author: podda
"""
import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import base64

def app():
    def csv_downloader(data, typ):
    	csvfile = data.to_csv(index=False, header=True)
    	b64 = base64.b64encode(csvfile.encode()).decode()
    	new_filename = "temp.csv"
    	st.markdown("#### Download File ###")
    	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here - {typ}</a>'
    	st.markdown(href,unsafe_allow_html=True)
       
    st.markdown("## Data Upload")

    st.markdown("### Upload a csv file for analysis.") 
    c1,c2 = st.beta_columns((3,1))
    c1.text("*NOTE*") #- Download the template and upload the data strictly according to the template. \nFor multivariate data, the first feature should be the target variable.\nThere's no limit on the independent features.\nMaximum of 1000 rows should be provided.")
    c1.text('1. Download the template and upload the data strictly according')
    c1.text('to the template.')
    c1.text('2. For multivariate data, the first feature should be the target')
    c1.text('variable.')
    c1.text('3. There is no limit on the independent features.')
    c1.text('4. Maximum of 1000 rows should be provided.')
    if c2.button('Template-Univariate'):
        tempu = pd.read_csv('data/tempu.csv')    
        csv_downloader(tempu, 'univariate')
    if c2.button('Template-Multivariate'):
        tempm = pd.read_csv('data/tempm.csv')    
        csv_downloader(tempm, 'multivariate')
    
    tstype = st.selectbox("Please select the type of Time Series", ["Univariate","Multivariate"])
    if tstype == 'Univariate':
        uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
        global data
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                print(e)
                data = pd.read_excel(uploaded_file)
    
        ''' Load the data and save the columns with categories as a dataframe. 
        This section also allows changes in the numerical and categorical columns. '''
            
        if st.button("Load and Submit Univariate Data"):
            if len(data.columns) > 1:
                st.error('Invalid Data Loaded-Please upload according to the template')
            else:
                st.dataframe(data)
                data = data.fillna(method='ffill').fillna(method='bfill')
                data.to_csv('data/udata.csv', index=False)
                st.success('Upload Successful')
    else:
        uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
        global data1
        if uploaded_file is not None:
            try:
                data1 = pd.read_csv(uploaded_file)
            except Exception as e:
                print(e)
                data1 = pd.read_excel(uploaded_file)
    
        ''' Load the data and save the columns with categories as a dataframe. 
        This section also allows changes in the numerical and categorical columns. '''
        
        if st.button("Load Data and Submit Multi-Variate Data"):
            st.dataframe(data1)
            data1 = data1.fillna(method='ffill').fillna(method='bfill')
            if len(data1.columns) > 5:
                xc = data1.drop(['y'], axis=1)
                num_components = 4
                pca = PCA(num_components)  
                xco = pca.fit_transform(xc)
                n_pcs= pca.n_components_ 
                most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
                initial_feature_names = xc.columns
                most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
                imp = ['y']
                imp = imp + most_important_names
                data_f = data1[imp]
                data_f.to_csv('data/mdata.csv', index=False)
            else:
                data1.to_csv('data/mdata.csv', index=False)
            st.success('Upload Successful')