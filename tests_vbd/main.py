import streamlit as st
import pandas as pd
import sys
sys.path.append("/home/virginie.desharnais@Digital-Grenoble.local/Documents/DataForGood-embeddings_clustering/DataForGood/OpenFoodFact/utils/")

import charts, upload_file_image
import plotly.express as px
from streamlit_plotly_events import plotly_events

import plotly.io as pio
pio.templates.default = "plotly"

tab1, tab2 = st.tabs(["📈 Chart", "📷 Image "])

# ---------------------------------- 1er Onglet Visualisation des clusters obtenus -------------------------------

tab1.subheader("Visualisation des clusters Objets / Etiquettes")
# tab1.write("Coucou le monde!")

df_chart = pd.DataFrame(
    {
    'blabla': [5, 10, 15, 20],
    '2_column': [100, 20, 30, 40],
    'size': [20, 25, 10, 15],
    'cluster': [0, 0, 1, 0]
    }
)


charts.show_scatter2D(df_chart, tab=tab1)

charts.show_scatter3D(df_chart, tab=tab1)

# ---------------------------------- 2ème Onglet Photo et plus proches voisins -------------------------------

# upload_file_image.upload_csv_file(tab=tab2)
tab2.subheader("Recherche de doublons parmi les images de la base de produits")
dir = "data/images/"
upload_file_image.upload_picture(tab=tab2, dir=dir)

# Appel du pipeline pour chercher les + proches voisins

# Affichage des + proches voisins
print(upload_file_image.list_images("data/image_proche/"))