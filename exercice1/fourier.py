import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("./../images/car.jpg",cv2.IMREAD_GRAYSCALE)

resized_img = cv2.resize(img,(1024, 576))


# Appliquer la transformation de Fourier
dft = cv2.dft(np.float32(resized_img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Calculer le spectre de magnitude
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Manipuler le spectre de fréquence
# Exemple : on applique un filtre passe-bas en masquant les hautes fréquences
rows, cols = resized_img.shape
crow, ccol = rows // 2 , cols // 2
# Créer un masque avec un carré central de 30x30 pixels, et mettre le reste à 0
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-15:crow+15, ccol-15:ccol+15] = 1

# Appliquer le masque au spectre décalé
fshift = dft_shift * mask

# Transformation inverse de Fourier
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Afficher les résultats
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Image originale')
plt.imshow(resized_img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Spectre de fréquence')
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Image transformée inverse')
plt.imshow(img_back, cmap='gray')

plt.show()