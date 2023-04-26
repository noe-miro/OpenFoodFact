import pickle
import numpy as np
from tensorflow.keras.utils import load_img

#generate embeddings
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

#dimension reduction
from sklearn.base import BaseEstimator, TransformerMixin
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



class Process_data(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._embeddings = Nonecd
        self._embeddings_light = None
        self._embeddings_proj3D = None
        self._reductor_100 = UMAP(n_components=100, random_state=42)
        self._reductor_3 = UMAP(n_components=3, random_state=42)
        self._filenames = None
        self._x_umap = None
        self._cosine_sim = None
        self._kmeans = KMeans(n_clusters=2, random_state=42)
        

    #----LOAD EMBEDDING FILE
    def load_emb(self, path):
        '''
        Load all embeddings from a path, generate 2 lists : filenames and feat
        use get_embeddings_raw or get_embeddings_light
        '''
        
        with open(path, 'rb') as f:
            emb = pickle.load(f)

        # get a list of the filenames
        filenames = np.array(list(emb.keys()))
        # get a list of the raw embeddings
        feat = np.array(list(emb.values()))
        feat = feat.reshape(-1,4096)
        self._filenames = filenames
        return feat

    def get_embeddings_raw(self, path):
        self._embeddings = self.load_emb(path)


    def get_embeddings_light(self, path):
        self._embeddings_light = self.load_emb(path)


    #----generate embeddings
    def extract_features(self, path_to_file):
        '''
        from one image, generate embeddings with VGG16 model
        '''
        
        model = VGG16()
        model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
        # load the image as a 224x224 array
        img = load_img(path_to_file, target_size=(224,224))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img) 
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        img = img.reshape(1,224,224,3) 
        # prepare image for model
        x = preprocess_input(img)
        # get the feature vector
        img_features = model.predict(x, use_multiprocessing=True)

        return img_features


    def fit_transform(self):
        '''
        from a list of raw embeddings, generate : light embeddings, reducer object, cosine matrix, kmeans
        '''

        #reduce dimension
        self._x_umap = self._reductor_100.fit_transform(self._embeddings)

        #cosine matrix
        self._cosine_sim = cosine_similarity(self._x_umap)

        #kmeans
        self._kmeans.fit(self._cosine_sim)

    def fit(self):
        pass
    
    def transform(self, path_to_file):
        '''
        return an image close to the first image
        '''
        
        img_emb = self.extract_features(path_to_file)
        img_u_map = self._reductor_100.transform(img_emb)

        # in img_cosine_sim there is a vector of length(ddataset) with the similarity to each image
        # TODO
        img_cosine_sim = cosine_similarity(self._x_umap, img_u_map)

        return img_cosine_sim


    #----PROJ3D
    def proj3D(self):
        '''
        from light embeddings, generate ultra light embeddings (projection in 2 or 3 dimensions)
        liste les images par cluster
        '''
        
        self._embeddings_proj3D = self._reductor_3.fit_transform(self._x_umap)

        #utile pour plotter
        xx0 = self._embeddings_proj3D[self._kmeans.labels_ == 0]
        xx1 = self._embeddings_proj3D[self._kmeans.labels_ == 1]

        return xx0, xx1


    #----ASSIGNATION AUTO
    def assignation(self):
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
        k = [k for k, v in zip(self._filenames, self._kmeans.labels_) for label in list_labels if label in v ]

        #on recuperer la valeur majoritaire de cette liste : c'est le numéro du cluster annoté donc les étiquettes
        unique, counts = np.unique(k, return_counts=True)
        bool_etiquette = unique[np.argmax(counts)]

        return bool_etiquette


    #----GENERATION DATAFRAME
    def generation_df(self, bool_etiquette):
        '''
        generate database for streamlit
        '''
        
        class_dict = {bool_etiquette:'etiquette',
                    not(bool_etiquette):'objet'}

        dataframe=[]
        for filename, label, emb in zip(self.filenames, self._kmeans.labels_, self._embeddings_proj3D):
            label_class = class_dict[str(label)]
            dataframe.append((filename, label, label_class, emb))
        
        return dataframe



