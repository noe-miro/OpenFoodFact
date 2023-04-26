import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import random as rd
import os

def upload_csv_file (tab):
    uploaded_file = st.file_uploader("Choose a CSV file")

    if uploaded_file is not None:
        df_file = pd.read_csv(uploaded_file, encoding = "iso-8859-1", sep=";")
        return st.dataframe(df_file)
    
def list_images(dir):
    list_file = list()
    for file in os.listdir(dir):
        list_file.append(file)
    return list_file
    
def upload_picture(dir):
    all_images = list_images(dir)
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)

        if image is not None:
            st.image(
                image,
                caption=f"Image chargée à tester\n" + img_file_buffer.name,
                width=100
            )
            

            with open(os.path.join("tmp/",img_file_buffer.name),"wb") as f: 
                f.write(img_file_buffer.getbuffer())         
                st.success("Saved File")

        return img_file_buffer.name

st.header("Recherche de doublons parmi les images de la base de produits")
dir = "data/images/"
file_name = upload_picture(dir=dir)

# Appel du pipeline pour chercher les + proches voisins

# Affichage des + proches voisins
# print(list_images("data/image_proche/"))