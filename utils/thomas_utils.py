import pickle
import numpy as np
from tensorflow.keras.utils import load_img

#generate embeddings
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

#dimension reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# clustering 
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

#affichage
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg



#----LOAD RAW
def load_raw(path):
    '''
    Load all embeddings raw from a path, generate 2 lists : filenames and feat
    '''
    
    with open(path, 'rb') as f:
        emb_raw = pickle.load(f)

    # get a list of the filenames
    filenames = np.array(list(emb_raw.keys()))
    # get a list of the raw embeddings
    feat = np.array(list(emb_raw.values()))
    feat = feat.reshape(-1,4096)

    return filenames, feat



#----generate embeddings
def extract_features(file):
    '''
    from one image, generate embeddings with VGG16 model
    '''
    
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    img = img.reshape(1,224,224,3) 
    # prepare image for model
    x = preprocess_input(img)
    # get the feature vector
    features = model.predict(x, use_multiprocessing=True)
    
    return features



#----REDUCE DIMENSION, COSINE AND KMEANS
def calculate(feat):
    '''
    from a list of raw embeddings, generate : light embeddings, reducer object, cosine matrix, kmeans
    '''

    #reduce dimension
    reducer = UMAP(n_components=100, random_state=42)
    x_umap = reducer.fit_transform(feat)

    #cosine matrix
    cosine_sim = cosine_similarity(x_umap)

    #kmeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(cosine_sim)

    return x_umap, reducer, cosine_sim, kmeans



#----PROJ3D
def proj3D(x_umap, kmeans):
    '''
    from light embeddings, generate ultra light embeddings (projection in 2 or 3 dimensions)
    liste les images par cluster
    '''
    
    umap3 = UMAP(n_components=3)
    xx = umap3.fit_transform(x_umap)

    #utile pour plotter
    xx0 = xx[kmeans.labels_ == 0]
    xx1 = xx[kmeans.labels_ == 1]

    return xx, xx0, xx1



#----ASSIGNATION AUTO
def assignation(filenames, kmeans):
    '''
    generate boolean for good cluster assignation 
    '''

    list_labels = ['0008295663177_2.400.jpg', 
                '0024000238119_14.400.jpg', 
                '0013600000745_4.400.jpg', 
                '0012000071744_7.400.jpg', 
                '0011110874245_2.400.jpg', 
                '0024000016854_3.400.jpg', 
                '0014100074120_13.400.jpg', 
                '0034000312733_5.400.jpg', 
                '0020601407893_2.400.jpg', 
                '0012000142451_7.400.jpg']

    # #list de key (donc cluster) pour chaque image annoté de list_label
    k = [k for k, v in zip(filenames, kmeans.labels_) for label in list_labels if label in v ]

    #on recuperer la valeur majoritaire de cette liste : c'est le numéro du cluster annoté donc les étiquettes
    unique, counts = np.unique(k, return_counts=True)
    bool_etiquette = unique[np.argmax(counts)]

    return bool_etiquette



#----GENERATION DATAFRAME
def generation_df(filenames, kmeans, xx):
    '''
    generate database for streamlit
    '''
    
    class_dict = {bool_etiquette:'etiquette',
                not(bool_etiquette):'objet'}

    dataframe=[]
    for filename, label, emb in zip(filenames, kmeans.labels_, xx):
        label_class = class_dict[str(label)]
        dataframe.append((filename, label, label_class, emb))
    
    return dataframe



