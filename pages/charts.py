import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_plotly_events import plotly_events

import plotly.io as pio
pio.templates.default = "plotly"

def show_scatter2D(df):

    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[1],
        color=df.columns[3], opacity=0.6, title='2D Representation of clusters Product VS Tag'
    )

    selected_points = plotly_events(fig)
    if selected_points:
        df_image = pd.DataFrame.from_dict(selected_points[0],orient='index')
        image_num = df_image.loc['pointNumber',0]
        image_x = df_image.loc['x',0]
        image_y = df_image.loc['y',0]
        
        st.write("index=" +  str(image_num) + "\nx=" + str(image_x) + "\ny=" + str(image_y))

def show_scatter3D(df):

    fig = px.scatter_3d(
        df, 
        x=df.columns[0], y=df.columns[1], z=df.columns[2],
        color=df.columns[3], opacity=0.6, title='3D Representation of clusters Product VS Tag'
    )
    
    selected_points = plotly_events(fig)
    if selected_points:
        df_image = pd.DataFrame.from_dict(selected_points[0],orient='index')
        image_num = df_image.loc['pointNumber',0]
        image_x = df_image.loc['x',0]
        image_y = df_image.loc['y',0]
        image_z = df_image.loc['curveNumber',0]
        
        st.write("index=" +  str(image_num) + "\nX=" + str(image_x) + "\nY=" + str(image_y)+ "\nZ=" + str(image_z)) 


df_chart = pd.DataFrame(
    {
    'blabla': [5, 10, 15, 20],
    '2_column': [100, 20, 30, 40],
    'size': [20, 25, 10, 15],
    'cluster': [0, 0, 1, 0]
    }
)
show_scatter2D(df_chart)
show_scatter3D(df_chart)