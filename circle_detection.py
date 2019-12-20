import cv2
import numpy as np
import time

def detect_circles(image):
    print(image.shape)
    cimg = image
    # cimg = cv2.pyrMeanShiftFiltering(image, 10, 100)
    cimg = cv2.blur(cimg, (5, 5)) # this makes circle more accurate
    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', cimg)

    # circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 400, param1=80, param2=10, minRadius=300, maxRadius=400)  #for bow, multi circles
    # circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 400, param1=80, param2=20, minRadius=300, maxRadius=400)  # for bowl，circle not accurate
    # circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 400, param1=120, param2=20, minRadius=300, maxRadius=400)  # for bowl，circle not accurate
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 400, param1=100, param2=20, minRadius=300, maxRadius=400)  # for bowl
    # circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 400, param1=50, param2=35, minRadius=20, maxRadius=40)  # for capstan, multi circles
    # circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 400, param1=50, param2=40, minRadius=20, maxRadius=40)  # for capstan
    if circles == None:
        print('no circle found')
        return

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0],i[1]), i[2], (0,0,255), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)
    # cv2.namedWindow('circles', cv2.WINDOW_NORMAL)
    # image = cv2.resize(image, (image.shape[0]//2, image.shape[1]//2))
    print(image.shape)
    cv2.imshow('circles', image)
    cv2.waitKey(0)


# use_camera = False
use_camera = True
if use_camera:
    capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture("/Users/zhouyou/Downloads/cv/vedio.mp4")
    # 获取 capture 的一些属性
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print('use camera: ({}, {}, {})'.format(frame_width, frame_height, fps))

    if capture.isOpened() is False:
        print('Error openning the camera')

    frame_index = 0
    while capture.isOpened():
        ret, img = capture.read()
        if not ret:
            break

        # 显示摄像头捕获的帧
        cv2.imshow('Input frame from the camera', img)

        # # 把摄像头捕捉到的帧转换为灰度
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # 显示处理后的帧
        # cv2.imshow('Grayscale input camera', gray_frame)

        detect_circles(img)

        # cv2.waitKey()这个函数是在一个给定的时间内(单位ms)等待用户按键触发
        # 如果用户没有按下按键，则继续等待(循环)
        if (cv2.waitKey(10) & 0xFF) == ord('q'):
            print('received q')
            break

        # if (cv2.waitKey(10) & 0xFF) == ord('c'):
        #     print('save')
        #     frame_name = 'camera_frame_{}.png'.format(frame_index)
        #     gray_frame_name = 'grayscale_camera_frame_{}.png'.format(frame_index)
        #     print(frame_name)
        #     cv2.imwrite(frame_name, frame)
        #     cv2.imwrite(gray_frame_name, gray_frame)
        #     frame_index += 1

        time.sleep(0.1)

    capture.release()
else:
    # img = cv2.imread("/Users/zhouyou/temp/circle.png")
    # img = cv2.imread("/Users/zhouyou/temp/capstan.jpeg")
    # img = cv2.imread("/Users/zhouyou/temp/bowl_3.jpeg")
    img = cv2.imread("/Users/zhouyou/temp/plate.jpeg")

    cv2.imshow('original image', img)
    # img = img[300:800, 300:800]
    # img = img[300:1050, 100:950]
    detect_circles(img)

    # cv2.rectangle(img, (300,300), (800,800), (0,0,255), 2)
    # cv2.circle(img, (550,550), 2, (0,0,255), 2)
    # cv2.imshow('rectangle', img)

    # result = cv2.blur(img, (5, 5))
    # cv2.imshow('circle detect 2', img)

    # canny = cv2.Canny(img, 100, 220)
    # cv2.imshow('canny', canny)

    cv2.waitKey(0)

cv2.destroyAllWindows()
exit(0)



