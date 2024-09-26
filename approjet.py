import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from data_processing import extract_features, process_datasets
from distances import retrieve_similar_images
import tempfile
import os
import matplotlib.pyplot as plt

# Charger les signatures des images
signatures = process_datasets('../Dataset')

def main(image, features, distance, signatures):
    # Rechercher des images similaires en fonction des caractéristiques de l'image téléchargée
    result = retrieve_similar_images(features_db=signatures, query_features=features, distance=distance.lower(), num_results=5)

    return result

# Définir le titre et les options dans la barre latérale
st.sidebar.title("Vous pouvez faire vos choix ici")
descriptor = st.sidebar.selectbox("Descripteur", ("GLCM", "Bitdesc","Haralick","Bitdesc+Haralick","GLCM+Haralick"))
distance = st.sidebar.selectbox("Distance", ("Euclidean", "Manhattan", "Chebyshev", "Canberra"))

# Titre principal de l'application
st.title("Projet IA: Affichage des images similaire")

# Charger les signatures des images
signatures = process_datasets('../Dataset')

# Section de téléchargement de l'image
uploaded_file = st.file_uploader("Uploader une image", type=[".jpg", ".png"])

# Si une image est téléchargée
if uploaded_file is not None:
    # Ouvrir et afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image Uploaded", use_column_width=True)
    # Ajouter ces lignes pour sauvegarder l'image dans un fichier temporaire
    temp_image_path = os.path.join(tempfile.gettempdir(), "uploaded_image.png")
    image.save(temp_image_path)

    # Extraire les caractéristiques de l'image en fonction du descripteur choisi
    features = extract_features(temp_image_path)

    # Afficher les caractéristiques de l'image
    st.subheader("Caractéristiques de l'image")
    df = pd.DataFrame(features, columns=["Valeur"])
    st.write(df)

    # Afficher un titre personnalisé pour la section de seuil de distance
    st.sidebar.markdown("**Seuil de Distance**")

    # Afficher un slider personnalisé pour le seuil de distance
    distance_threshold = st.sidebar.slider("Choisir le Seuil de Distance", 0, 800, (0, 800), step=10, help="Utilisez le slider pour ajuster le seuil de distance")

    # Rechercher des images similaires en fonction des caractéristiques de l'image téléchargée
    result = main(image, features, distance, signatures)

    # Afficher les résultats de la recherche
    st.write("Résultats de la recherche : ")
    for img_path, _, _ in result:
        similar_image = Image.open(img_path)
        st.image(similar_image, caption="Similar Image")
    
    # Afficher un histogramme du nombre d'images similaires
    num_similar_images = len(result)
    plt.bar(["Images similaires"], [num_similar_images])
    plt.xlabel("Type")
    plt.ylabel("Nombre d'images")
    st.pyplot(plt)
