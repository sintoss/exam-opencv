import cv2

def object_tracking():
    # Initialiser la capture vidéo (0 pour la webcam, ou le chemin d'un fichier vidéo)
    cap = cv2.VideoCapture(0)

    # Vérifier si la capture est ouverte correctement
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo")
        return

    # Lire la première frame
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire la vidéo")
        return

    # Sélectionner la ROI manuellement
    bbox = cv2.selectROI("Sélectionnez l'objet à suivre", frame, fromCenter=False, showCrosshair=True)

    # Initialiser le tracker KCF
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bbox)

    while True:
        # Lire une nouvelle frame
        ret, frame = cap.read()
        if not ret:
            break

        # Mettre à jour le tracker
        success, bbox = tracker.update(frame)

        if success:
            # Le suivi a réussi, dessiner le rectangle
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Suivi", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Le suivi a échoué
            cv2.putText(frame, "Perdu", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Afficher le résultat
        cv2.imshow("Suivi d'objet", frame)

        # Sortir de la boucle si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

# Lancer le suivi d'objet
object_tracking()