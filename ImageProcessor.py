import cv2 as cv
import numpy as np


def pre_process(img):
    img = cv.flip(img, 1)
    return img


def face_delete(img):
    detector = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img[y:y + h, x:x + w, :] = 0
    return img


def YCrCb_Otsu_detect(img):
    imgYUC = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    sImg = cv.split(imgYUC)
    outputMask = sImg[1]
    _, binImg = cv.threshold(outputMask, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    img = cv.copyTo(img, binImg)
    return img


def hand_partition(img):
    hand = face_delete(img)
    hand = YCrCb_Otsu_detect(hand)
    kernel = np.ones([18, 5], np.uint8)
    hand = cv.erode(hand, kernel, iterations=1)
    hand = cv.dilate(hand, kernel, iterations=1)
    hand = cv.erode(hand, kernel, iterations=1)
    hand = cv.dilate(hand, kernel, iterations=1)
    hand = cv.erode(hand, kernel, iterations=1)
    hand = cv.dilate(hand, kernel, iterations=1)
    return hand


def rock_sissor_paper_detect(img):
    hand = hand_partition(img)

    return

