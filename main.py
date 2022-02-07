
import numpy as np
import cv2
import os
import cv2.aruco as aruco
# Modifier ICI L'ADDRESSE IP
addresse_ip = '192.168.184.171:4747'

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

flux = 'http://'+addresse_ip+'/videostream.cgi'
webcam = cv2.VideoCapture(flux)

# Boucle d'affichage infinie jusqu'a arret complet
while (1):

    #Lecture des successions d'images depuis la camera
    _, imageFrame = webcam.read()

    # Conversion de Rgb
    # BGR(RGB color space) à
    # HSV(hue-saturation-value)
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Paramètre du masque rouge
    # definition du masque
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    # red_lower = np.array([156, 179, 158], np.uint8)
    # red_upper = np.array([179, 227, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)


    # Paramètre du masque vert
    # definition du masque
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    # green_lower = np.array([30, 109, 137], np.uint8)
    # green_upper = np.array([65, 155, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Paramètre du masque bleu
    # definition du masque
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    # blue_lower = np.array([123, 60, 69], np.uint8)
    # blue_upper = np.array([163, 121, 111], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    #transfo morphologique
    kernal = np.ones((5, 5), "uint8")

    # Pour la couleur ROUGE
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # Pour la couleur VERTE
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)

    # Pour la couleur Bleue
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=blue_mask)

    #Creation des contours pour afficher la couleur en réel
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, "Rouge", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

            # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 255, 0), 2)

            cv2.putText(imageFrame, "Vert", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0))

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)

            cv2.putText(imageFrame, "Bleu", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imageFrame, dictionary)  # Détecter le marqueur
    aruco.drawDetectedMarkers(imageFrame, corners, ids, (0, 255, 0))  # Dessiner sur le marqueur détecté

    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break