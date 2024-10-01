import cv2
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
img = cv2.imread("./../images/car.jpg",cv2.IMREAD_GRAYSCALE)

resized_img = cv2.resize(img,(1024, 576))

# Vérifier si l'image est chargée correctement
if resized_img is None:
    print("Erreur lors du chargement de l'image")
    exit()

# Appliquer le seuillage adaptatif
# cv2.ADAPTIVE_THRESH_MEAN_C : Calculer le seuil en fonction de la moyenne des voisins
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C : Calculer le seuil en fonction de la moyenne pondérée des voisins
# 11 : Taille du bloc (taille de la région autour du pixel pour calculer le seuil)
# 2 : Constante soustraite de la moyenne ou moyenne pondérée pour obtenir le seuil final
thresh_mean = cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh_gaussian = cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Afficher les résultats
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Image originale')
plt.imshow(resized_img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Seuillage adaptatif (Mean)')
plt.imshow(thresh_mean, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Seuillage adaptatif (Gaussian)')
plt.imshow(thresh_gaussian, cmap='gray')


plt.show()