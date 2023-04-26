import streamlit as st
import pandas as pd
import sys
sys.path.append("/home/virginie.desharnais@Digital-Grenoble.local/Documents/DataForGood-embeddings_clustering/DataForGood/OpenFoodFact/utils/")

import charts, upload_file_image

tab1, tab2 = st.tabs(["📈 Chart", "📷 Image "])

# ---------------------------------- 1er Onglet Visualisation des clusters obtenus -------------------------------

tab1.subheader("Visualisation des clusters Objets / Etiquettes")
# tab1.write("Coucou le monde!")

df_chart = pd.DataFrame(
    {
    'blabla': [5, 10, 15, 20],
    '2_column': [100, 20, 30, 40],
    'size': [20, 25, 10, 15]
    }
)

charts.show_scatter(df_chart, "streamlit", tab=tab1)

# ---------------------------------- 2ème Onglet Photo et plus proches voisins -------------------------------

# upload_file_image.upload_csv_file(tab=tab2)
tab2.subheader("Recherche de doublons parmi les images de la base de produits")
dir = "data/images/"
upload_file_image.upload_picture(tab=tab2, dir=dir)

print(upload_file_image.list_images("data/etiquette_sample/"))