import cv2, numpy as np, os
from Modules.HandTracking import handTrackingModule as htm

###############################################
xp, yp = 0, 0
color = (0, 0, 0)
brush_thickness = 15
eraser_thickness = 30
###############################################
images = []
folder = "img"
name_img = os.listdir(folder)
img_cover = np.zeros((480, 640, 3), np.uint8)

for img in name_img:
    image = cv2.imread(f"{folder}/{img}")
    images.append(image)

header = images[1]
###############################################
cap = cv2.VideoCapture(1)

detector = htm.handDetector()
while True:
    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. find hand landmark
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    #
    if len(lmList) != 0:
        x1, y1 = lmList[0][1][8][1:]
        x2, y2 = lmList[0][1][12][1:]
        # 3. check which fingers are up
        h = detector.fingersUp(img, reverse=True)[1]

        # 4. selection mode
        if h[0][1][1] and h[0][1][2]:
            xp, yp = 0, 0
            cv2.line(img, (x1, y1), (x2, y2), color, 10)
            if y1 < 105:
                # print('selection mode')
                if 20 < x1 < 110:
                    # print("c")
                    header = images[1]
                    color = (0, 0, 0)
                elif 160 < x1 < 250:
                    # print("green")
                    header = images[2]
                    color = (0, 255, 0)
                elif 300 < x1 < 390:
                    # print("pink")
                    header = images[3]
                    color = (255, 0, 255)
                elif 440 < x1 < 530:
                    # print("blue")
                    header = images[0]
                    color = (255, 0, 0)

        # 5. drawing mode
        if h[0][1][1] and h[0][1][2] == 0:
            if color == (0, 0, 0):
                cv2.circle(img, (x1, y1), eraser_thickness, color, cv2.FILLED)
            else:
                cv2.circle(img, (x1, y1), brush_thickness, color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (0, 0, 0):
                cv2.line(img_cover, (xp, yp), (x1, y1), color, eraser_thickness)
            else:
                cv2.line(img_cover, (xp, yp), (x1, y1), color, brush_thickness)
            xp, yp = x1, y1

    img_gray = cv2.cvtColor(img_cover, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_cover)

    img[:105, :640] = header
    cv2.imshow('img', img)
    # cv2.imshow('cover', img_cover)
    # cv2.imshow('gray', img_gray)
    # cv2.imshow('inv', img_inv)
    cv2.waitKey(1)
