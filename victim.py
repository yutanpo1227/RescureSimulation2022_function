import cv2
import numpy as np

img = cv2.imread(
    #file path
    )

def victim_match(img1):
    max_val = 0
    max = []
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        x, y, width, height = cv2.boundingRect(cnt)
        area = width * height
        if max_val < area and x != 0 and y != 0:
            max_val = area
            max = [x,y,width,height]
    if len(max) = 0:
        return('T')
    cut_img = thresh[max[1] : max[1]+max[3], int(max[0]+max[2]/3) : int(max[0]+max[2]*2/3)]
    contours2, hierarchy = cv2.findContours(cut_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours2)>2):
        return('S')
    else:
        h,w = cut_img.shape[:2]
        i = 0
        while i < h:
            if cut_img[i,int(w/2)] < 127:
                y_line = i
                break
            i = i + 1
        if(y_line < h * 2/3):
            return('H')
        else:
            return('U')


def create_image(x):
    blank = np.zeros((200, 200, 3))
    blank += 255

    if x == 'H':
        cv2.putText(blank,
                    text=x,
                    org=(0, 190),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=10.0,
                    color=(0, 0, 0),
                    thickness=16,
                    lineType=cv2.LINE_AA)
    else:
        cv2.putText(blank,
                    text=x,
                    org=(0, 190),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=10.0,
                    color=(0, 0, 0),
                    thickness=14,
                    lineType=cv2.LINE_AA)
    blank = cv2.blur(src=blank, ksize=(5, 5))
    img = blank.astype(np.float32)
    return img


def get_var_name(var):
    for k, v in globals().items():
        if id(v) == id(var):
            name = k
    return name


def contour_point(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def victim_or_hazard(img):
    point = contour_point(img)
    if(point > 10):
        result = detect_hazard(img)
    else:
        result = victim_match(img)
    return result


def detect_red_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_min = np.array([0, 64, 0])
    hsv_max = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    hsv_min = np.array([150, 64, 0])
    hsv_max = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    mask = mask1 + mask2

    return mask


def detect_yellow_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_min = np.array([20, 80, 10])
    hsv_max = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    approx_contours = []
    for i, cnt in enumerate(contours):
        arclen = cv2.arcLength(cnt, True)
        print(len(cnt))

    return mask


def calc_black_whiteArea(bw_image):
    image_size = bw_image.size
    whitePixels = cv2.countNonZero(bw_image)
    blackPixels = bw_image.size - whitePixels

    whiteAreaRatio = (whitePixels / image_size) * 100
    blackAreaRatio = (blackPixels / image_size) * 100

    return whiteAreaRatio


def detect_hazard(img):

    red_mask = detect_red_color(img)
    yellow_mask = detect_yellow_color(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([0, 0, 100])
    hsv_max = np.array([180, 45, 255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    red_per = calc_black_whiteArea(red_mask)
    yellow_per = calc_black_whiteArea(yellow_mask)
    dst = cv2.cornerHarris(yellow_mask, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    if yellow_per > 10:
        if len(dst) == 3:
            return 'O'
        elif red_per > 20:
            return 'F'
        else:
            white_per = calc_black_whiteArea(mask)
            if white_per > 84:
                return 'P'

    elif red_per > 20:
        return 'F'
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_min = np.array([0, 0, 100])
        hsv_max = np.array([180, 45, 255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        white_per = calc_black_whiteArea(mask)
        if white_per > 84:
            return 'P'

    return 'C'


def checkVic(img):
    img = np.frombuffer(img, np.uint8).reshape(
        (cam.getHeight(), cam.getWidth(), 4))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img, thresh = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        contArea = cv.contourArea(cnt)
        ratio = w / h
        if contArea > 300 and contArea < 1000 and ratio > 0.65 and ratio < 0.95:
            return True
    return False


print(victim_match(img))
