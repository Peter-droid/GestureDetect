import cv2 as cv
import numpy as np


def pre_process(img):
    img = cv.flip(img, 1)
    return img


def face_delete(img):
    detector = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    img_without_face = np.array([]);
    img_without_face = img.copy()
    for (x, y, w, h) in faces:
        img_without_face[y - 20:y + h + 50, x - 10:x + w + 10, :] = 0
    return img_without_face


def YCrCb_Otsu_detect(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    # 图像分割, 分别获取y, cr, br通道图像
    (y, cr, cb) = cv.split(ycrcb)
    # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
    # 对cr通道分量进行高斯滤波
    cr1 = cv.GaussianBlur(cr, (5, 5), 0)
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    _, skin1 = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    img = cv.copyTo(img, skin1)
    return img


def hand_partition(img):
    hand = face_delete(img)
    hand = YCrCb_Otsu_detect(hand)
    kernel1 = np.ones([10, 5])
    kernel2 = np.ones([5, 10])
    kernel3 = np.ones([3, 3])
    hand = cv.erode(hand, kernel1)
    # hand = cv.erode(hand, kernel2)
    hand = cv.dilate(hand, kernel1)
    # hand = cv.dilate(hand, kernel2)
    hand = cv.erode(hand, kernel1)
    # hand = cv.erode(hand, kernel2)
    hand = cv.dilate(hand, kernel1)
    # hand = cv.dilate(hand, kernel2)
    hand = cv.erode(hand, kernel3)
    return hand


def hand_contour(hand):
    edge = cv.Canny(hand, 50, 200)
    contours, hierachy = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv.contourArea(contours[i]))
    area = np.array(area)
    hand_index = []
    for i in range(len(contours)):
        if area[i] > 4500:
            hand_index.append(i)
    return contours, hand_index


def rock_sissor_paper_detect(img, contour):
    cont_s = cv.contourArea(contour)
    hull = cv.convexHull(contour, clockwise=True)
    convex_s = cv.contourArea(hull)
    area_k = convex_s / cont_s
    if area_k <= 1.2:
        Str = "Rock"
    elif area_k < 1.6:
        Str = "Scissor"
    else:
        Str = "Paper"
    cv.putText(img, Str, hull[0][0], cv.FONT_HERSHEY_TRIPLEX, 0.8, [255, 0, 0], 2)
    img = cv.drawContours(img, [contour], 0, [0, 0, 0], 2)
    img = cv.drawContours(img, [hull], 0, [0, 255, 255], 3)
    return img


def number_detect(img, contour):
    hull = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull)
    cnt = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            # s,e,f其实是轮廓点集的下标
            start = tuple(contour[s][0])  # 得到x,y坐标
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            # 求出欧氏距离
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            # 余弦定理
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            if angle <= np.pi / 2:
                cnt += 1
    if cnt == 0:
        Str = '1'
    elif cnt == 1:
        Str = '2'
    elif cnt == 2:
        Str = '3'
    elif cnt == 3:
        Str = '4'
    elif cnt == 4:
        Str = '5'
    else:
        Str = "null"
    hull = cv.convexHull(contour)
    cv.putText(img, Str, hull[0][0], cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv.drawContours(img, [contour], 0, [0, 0, 0], 2)
    cv.drawContours(img, [hull], 0, [255, 255, 255], 3)
    return img


def others_detect(img, contour):
    cont_s = cv.contourArea(contour)
    hull = cv.convexHull(contour, clockwise=True)
    convex_s = cv.contourArea(hull)
    area_k = convex_s / cont_s
    print(area_k)
    if 1.4 > area_k > 1.2:
        Str = "OK"
    elif 1.6 > area_k > 1.4:
        Str = "Yeah"
    elif area_k <= 1.2:
        Str = "Good"
    elif 2 > area_k > 1.6:
        Str = "Rock"
    else:
        Str = "Null"

    cv.putText(img, Str, hull[0][0], cv.FONT_HERSHEY_TRIPLEX, 0.8, [255, 0, 0], 2)
    img = cv.drawContours(img, [contour], 0, [0, 0, 0], 2)
    img = cv.drawContours(img, [hull], 0, [0, 255, 0], 3)
    return img


def detect_status(img, status):
    point = [15, 25]
    frame = pre_process(img)
    hand = hand_partition(frame)
    contours, hand_index = hand_contour(hand)
    if status == 0:
        func = rock_sissor_paper_detect
        Str = "Rock Paper Scissors"
    elif status == 1:
        func = number_detect
        Str = "Number"
    else:
        func = others_detect
        Str = "Others"
    for i in range(len(hand_index)):
        contour = contours[hand_index[i]]
        frame = func(frame, contour)
    cv.putText(frame, Str, point, cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    return frame


def detect():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_EXPOSURE, -7.7)
    status = 0
    while 1:
        _, frame = cap.read()
        frame = detect_status(frame, status)
        cv.namedWindow("image", 0)
        cv.resizeWindow("image", 640, 480)
        cv.imshow("image", frame)
        k = cv.waitKey(5)
        if k == 32:
            status = status + 1
            if status == 3:
                status = 0
        if k == 27:
            break
    cv.destroyAllWindows()

