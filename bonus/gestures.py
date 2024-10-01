import cv2

# Charger les classifieurs en cascade pour les mains ouvertes et fermées
open_hand_cascade = cv2.CascadeClassifier('palm.xml')
closed_hand_cascade = cv2.CascadeClassifier('fist.xml')

# Ouvrir la caméra
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les mains ouvertes et fermées
    open_hands = open_hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    closed_hands = closed_hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Dessiner des rectangles autour des mains détectées
    for (x, y, w, h) in open_hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Main ouverte", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    for (x, y, w, h) in closed_hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Main fermee", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Afficher la vidéo avec les mains détectées
    cv2.imshow('Video - Mains détectées', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
