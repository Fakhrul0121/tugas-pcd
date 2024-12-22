import argparse
import cv2

def detect_edge(filename):
    img = cv2.imread(filename)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(img_gray, (3,3), 0)

    canny = cv2.Canny(blur_img, 100, 200)

    #invert edge_image
    canny = cv2.bitwise_not(canny)

    cv2.imwrite("edge.png", canny)

    print("edge detection complete")

