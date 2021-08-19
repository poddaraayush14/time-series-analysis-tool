# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:25:34 2021

@author: podda
"""
import os
import streamlit as st
import numpy as np
from PIL import  Image


# Custom imports 
from multipage import MultiPage
from pages import data_upload, forecast, data_viz

#st.set_page_config(layout="wide")

 
app = MultiPage()

#col1, col2 = st.beta_columns((2,1))
display = Image.open('Logo_f.PNG')
display = np.array(display)
#col1, col2 = st.beta_columns(2)
st.image(display, width = 750)

#col2.title("Time Series Analysis Application")

app.add_page("Upload Data", data_upload.app)
app.add_page("Data Visualization and Analysis",data_viz.app)
app.add_page("Forecast Time Series", forecast.app)


# The main app
app.run()








