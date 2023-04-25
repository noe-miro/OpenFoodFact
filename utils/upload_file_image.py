import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import random as rd
import os

def upload_csv_file (tab):
    uploaded_file = tab.file_uploader("Choose a CSV file")

    if uploaded_file is not None:
        df_file = pd.read_csv(uploaded_file, encoding = "iso-8859-1", sep=";")
        return tab.dataframe(df_file)
    
def list_images(dir):
    list_file = list()
    for file in os.listdir(dir):
        list_file.append(file)
    return list_file
    
def upload_picture(tab, dir):
    all_images = list_images(dir)
    img_file_buffer = tab.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)

        if image is not None:
            tab.image(
                image,
                caption=f"Image chargée à tester",
                width=100
            )
            list_img=rd.sample(all_images, 5)
            for index, x in enumerate(list_img):
                list_img[index] = "data/images/" + x

            tab.image(
                image=list_img,
                caption="plus proche voisin...",
                width=100
            )

