import cv2
import numpy as np

# Charger l'image
img = cv2.imread("./../images/car.jpg")
image = cv2.resize(img, (1024, 576))

# Définir les coordonnées de la région d'intérêt (ROI)
top_left = (672, 322)
bottom_right = (738, 336)

# Extraire la ROI
roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# Augmenter la luminosité de la ROI
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hsv_roi[..., 2] = cv2.add(hsv_roi[..., 2], 50)  # Augmenter le canal de valeur
roi_lum = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)

# Remettre la ROI modifiée dans l'image
image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi_lum

# Créer un masque pour le reste de l'image
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, top_left, bottom_right, 255, -1)  # Rectangle blanc dans le masque

# Flouter l'ensemble de l'image
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# Combiner l'image floue avec l'image originale en utilisant le masque
final_image = np.where(mask[:, :, np.newaxis] == 255, image, blurred_image)

# Afficher l'image finale
cv2.imshow('Image Finale', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sauvegarder l'image finale si nécessaire
cv2.imwrite('output_image.jpg', final_image)  # Remplacer par le chemin de sortie souhaité
