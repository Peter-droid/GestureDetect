import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
from ImageProcessor import *

#
#

#
#     # cv.namedWindow("image", 0)
#     # cv.resizeWindow("image", 640, 480)
#     # cv.imshow("image", img)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()
#
# # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_EXPOSURE, -7.7)
    while 1:
        _, frame = cap.read()
        frame = pre_process(frame)
        hand = hand_partition(frame)
#         kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, [5, 5])
#         dst = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
#         edge = cv.Canny(dst, 50, 200)
#         contours, hierachy = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         area = []
#         for i in range(len(contours)):
#             area.append(cv.contourArea(contours[i]))
#         maxPosition = -1
#         for i in range(len(area)):
#             if area[i] == max(area):
#                 maxPosition = i
#                 break
#
#         cont_s = area[maxPosition]
#         if cont_s >= 2:
#             hull = cv.convexHull(contours[maxPosition], clockwise=True)
#             frame = cv.drawContours(frame, [hull], 0, [255, 255, 255], 3)
#             convex_s = cv.contourArea(hull)
#             area_k = convex_s / cont_s
#             print(cont_s, area_k)
#             arc_k = cv.arcLength(contours[maxPosition], True) / cv.arcLength(hull, True)
#             rects = cv.boundingRect(contours[maxPosition])
#             cv.rectangle(frame, rects, [255, 255, 255], 2)
#             if area_k <= 1.2:
#                 cv.putText(frame, "Rock", [10, 20], cv.FONT_HERSHEY_TRIPLEX, 0.8, [255, 200, 200], 2)
#             elif area_k < 1.6:
#                 cv.putText(frame, "Scissors", [10, 20], cv.FONT_HERSHEY_TRIPLEX, 0.8, [255, 200, 200], 2)
#             else :
#                 cv.putText(frame, "Paper", [10, 20], cv.FONT_HERSHEY_TRIPLEX, 0.8, [255, 200, 200], 2)
#
        cv.namedWindow("image", 0)
        cv.resizeWindow("image", 640, 480)
        cv.imshow("image", frame)
        k = cv.waitKey(5)
        if k == 27:
            break
    cv.destroyAllWindows()




