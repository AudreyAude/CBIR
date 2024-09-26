# Scikit-Image
from skimage.feature import graycomatrix, graycoprops
# Import Bitdesc
from BiT import bio_taxo
import cv2

def glcm(data):
    glcm = graycomatrix(data, [2], [0],None, symmetric=True,normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]
    return [diss, cont, corr, ener, homo]

def Bitdesc(data):
    return bio_taxo(data)

# Fonction pour extraire les caractéristiques haralick
def Haralick(img):
    data = cv2.imread(img)  # Chargement de l'image en couleur
    gray_data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)  # Convertir l'image en niveaux de gris
    glcm_features = glcm(gray_data)
    return glcm_features

