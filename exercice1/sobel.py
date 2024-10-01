import cv2
import numpy as np

img = cv2.imread("./../images/car.jpg")

resized_img = cv2.resize(img,(1024, 576))

if resized_img is None:
    print("Erreur : Impossible de charger l'image")
else:
    # Appliquer le filtre de Sobel pour les contours verticaux
    sobelx = cv2.Sobel(resized_img, cv2.CV_64F, 1, 0, ksize=3)

    # Appliquer le filtre de Sobel pour les contours horizontaux
    sobely = cv2.Sobel(resized_img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculer la magnitude du gradient
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normaliser la magnitude pour l'affichage
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Afficher les résultats
    cv2.imshow('Image originale', resized_img)
    #cv2.imshow('Contours verticaux (Sobel X)', cv2.convertScaleAbs(sobelx))
    #cv2.imshow('Contours horizontaux (Sobel Y)', cv2.convertScaleAbs(sobely))
    cv2.imshow('Magnitude du gradient', magnitude)

    cv2.imwrite("./results/sobel.jpg",magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()