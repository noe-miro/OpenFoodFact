import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_plotly_events import plotly_events

def show_scatter(df, theme, tab):

    # Scatter plot with MatplotLib
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)

    # plt.scatter(x=map_data['lon'], y=map_data['lat'])

    # st.write(fig)


    fig = px.scatter(
        df,
        x=df.columns[0],
        y=df.columns[1]
    )
    selected_points = plotly_events(fig)
    if selected_points:
        df_image = pd.DataFrame.from_dict(selected_points[0],orient='index')
        image_num = df_image.loc['pointIndex',0]
        image_x = df_image.loc['x',0]
        image_y = df_image.loc['y',0]
        tab.plotly_chart(fig, theme=theme, use_container_width=True)
        tab.write("index=" +  str(image_num) + "\nx=" + str(image_x) + "\ny=" + str(image_y)) #, tab.dataframe(df_image)
    
    # else:
    #     tab.plotly_chart(fig, theme=theme, use_container_width=True)
