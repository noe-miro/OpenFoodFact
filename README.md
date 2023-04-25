# OpenFoodFact

OpenFoodFact is an open-source dataset which contains millions of images of food products and their labels as uploaded by customers around the world.

This project aims to provide ready-to-use tools to clean up the dataset of OpenFoodFact. The related issue can be found here: 
https://github.com/openfoodfacts/openfoodfacts-ai/issues/203

Users can upload as many photos as they like, and they might contain the product, the label in any language, or both. Some products are duplicated, and some images might be taken twice in the same upload as well.

In order to delete duplicates, one needs to know which images are very similar to a given one. An unsupervised approach for this task is clustering.

In a first step, we want to separate the labels from the products through clustering.
