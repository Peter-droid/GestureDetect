import cv2 as cv
import numpy as np


def get_camera():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_BRIGHTNESS, 50)     # 亮度 50
    cap.set(cv.CAP_PROP_CONTRAST, 100)      # 对比度 100
    cap.set(cv.CAP_PROP_SATURATION, 80)     # 饱和度 80
    cap.set(cv.CAP_PROP_HUE, 0)             # 色调 0
    cap.set(cv.CAP_PROP_EXPOSURE, -5.5)     # 曝光 -5.5
    return cap


def pre_process(img):
    img = cv.flip(img, 1)
    return img


def face_delete(img):
    # 获取分类器
    detector = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
    # 转化图像为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 利用分类器找出图中的脸部
    faces = detector.detectMultiScale(gray, 1.3, 5)
    # 建立脸部图像副本
    img_without_face = img.copy()
    for (x, y, w, h) in faces:
        # 将脸部区域灰度值归零
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
    # 删除脸部
    hand = face_delete(img)
    # 对Cr分量二值化
    hand = YCrCb_Otsu_detect(hand)
    # 形态学运算
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
    # 获取边缘
    edge = cv.Canny(hand, 50, 200)
    # 边缘点的坐标（矢量边缘）
    contours, hierachy = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    area = []
    # 依次计算初步分割后的图像中包含区域的面积
    for i in range(len(contours)):
        area.append(cv.contourArea(contours[i]))
    area = np.array(area)
    hand_index = []
    # 筛选出面积大于 4500 小于 50000 的区域
    for i in range(len(contours)):
        if 50000 > area[i] > 4500:
            hand_index.append(i)
    return contours, hand_index


def rock_sissor_paper_detect(img, contour):
    # 获得手部区域面积
    cont_s = cv.contourArea(contour)
    # 得到手部区域的凸包
    hull = cv.convexHull(contour, clockwise=True)
    # 计算凸包的面积
    convex_s = cv.contourArea(hull)
    # 计算凸包面积与手部区域面积的比值
    area_k = convex_s / cont_s
    # 若比值小于 1.2 ，则为“石头”
    if area_k <= 1.2:
        Str = "Rock"
    # 若比值大于 1.2 小于 1.6，则为“剪刀”
    elif area_k < 1.6:
        Str = "Scissor"
    # 若比值大于 1.6 则为“布”
    else:
        Str = "Paper"
    # 打印结果，描出区域和凸包
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
    # 根据缺陷中锐角的数量判断手势结果
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
    # 打印识别结果、轮廓和凸包
    cv.putText(img, Str, hull[0][0], cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv.drawContours(img, [contour], 0, [0, 0, 0], 2)
    cv.drawContours(img, [hull], 0, [255, 255, 0], 3)
    return img


def others_detect(img, contour):
    # 计算手部区域面积
    cont_s = cv.contourArea(contour)
    # 得到凸包
    hull = cv.convexHull(contour, clockwise=True)
    # 凸包面积
    convex_s = cv.contourArea(hull)
    # 凸包面积和手部区域面积的比值
    area_k = convex_s / cont_s
    # 根据不同的比值判断手势
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
    # 打印手势结果
    cv.putText(img, Str, hull[0][0], cv.FONT_HERSHEY_TRIPLEX, 0.8, [255, 0, 0], 2)
    img = cv.drawContours(img, [contour], 0, [0, 0, 0], 2)
    img = cv.drawContours(img, [hull], 0, [0, 255, 0], 3)
    return img


def detect_status(img, status):
    # 标题打印位置
    point = [15, 25]
    # 翻转
    frame = pre_process(img)
    # 获取手部图像
    hand = hand_partition(frame)
    # 获取手部轮廓
    contours, hand_index = hand_contour(hand)
    # 根据状态码筛选探测函数
    if status == 1:
        func = rock_sissor_paper_detect
        Str = "Rock Paper Scissors"
    elif status == 0:
        func = number_detect
        Str = "Number"
    else:
        func = others_detect
        Str = "Others"
    # 对每个检测到的手部轮廓进行检测
    for i in range(len(hand_index)):
        contour = contours[hand_index[i]]
        frame = func(frame, contour)
    # 打印检测的标题
    cv.putText(frame, Str, point, cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    return frame


def detect():
    # 获取摄像头
    cap = get_camera()
    # 初始化状态码
    status = 0
    while 1:
        # 读取一帧
        _, frame = cap.read()
        # 根据状态码进行手势检测
        frame = detect_status(frame, status)
        # 显示图片
        cv.namedWindow("image", 0)
        cv.resizeWindow("image", 640, 480)
        cv.imshow("image", frame)
        # 读取键盘
        k = cv.waitKey(5)
        # 若是空格，状态码+1
        if k == 32:
            status = status + 1
            if status == 3:
                status = 0
        # 若是 esc ，退出循环
        if k == 27:
            break
    cv.destroyAllWindows()

