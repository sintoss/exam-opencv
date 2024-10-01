import cv2

def highlightFace(net, frame, conf_threshold=0.7):
    # Copier le cadre d'origine pour le traitement
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Préparer l'image pour le modèle
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)

    # Effectuer la détection des visages
    detections = net.forward()
    faceBoxes = []

    # Parcourir les détections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:  # Vérifier si la confiance dépasse le seuil
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])

            # Dessiner un rectangle autour du visage détecté
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)

    return frameOpencvDnn, faceBoxes

# Chargement des modèles
faceProto = "./models/opencv_face_detector.pbtxt"
faceModel = "./models/opencv_face_detector_uint8.pb"
ageProto = "./models/age_deploy.prototxt"
ageModel = "./models/age_net.caffemodel"
genderProto = "./models/gender_deploy.prototxt"
genderModel = "./models/gender_net.caffemodel"

# Valeurs moyennes du modèle
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Homme', 'Femme']

# Charger les réseaux de détection
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Ouvrir la caméra
video = cv2.VideoCapture(0)
padding = 20

# Lire le premier cadre
hasFrame, frame = video.read()
if not hasFrame:
    print("Impossible de lire un cadre de la vidéo.")
    exit()

# Redimensionner la fenêtre de sortie
desired_window_width = 480
ratio = desired_window_width / frame.shape[1]
height = int(frame.shape[0] * ratio)

# Boucle principale pour la détection
while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        break

    # Redimensionner le cadre
    frame = cv2.resize(frame, (desired_window_width, height))

    # Mettre en surbrillance le visage
    resultImg, faceBoxes = highlightFace(faceNet, frame)

    # Traiter chaque visage détecté
    for faceBox in faceBoxes:
        face = frame[
            max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
            max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)
        ]

        # Préparer le visage pour la détection du genre
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Préparer le visage pour la détection de l'âge
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Afficher le texte du genre et de l'âge sur l'image
        text = "{}:{}".format(gender, age)
        cv2.putText(resultImg, text, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Détection d'âge et de genre", resultImg)

    # Sortie de la boucle si 'q' est pressé
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Libérer les ressources
cv2.destroyAllWindows()
video.release()
