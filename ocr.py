import cv2
import numpy as np
import pytesseract
import re


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# đọc ảnh và tiền xử lý ảnh
img = cv2.imread("test1.jpg")
gray = get_grayscale(img)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\\tesseract.exe"
char = 'qewrytiuopadfghkjlzcvxbnm'

#custom_config = r'--oem 3 --psm 6 outputbase digits'
#custom_config2 = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQSRTUVWXYZ --psm 6'
text = pytesseract.image_to_string(thresh,lang= "eng")
print("Nhân dạng văn bản :")
print(text)

text = re.sub('\n', ' ', text)
text = re.sub('\s+', ' ', text)
#lọc chữ số
number = '\d+'
findnum = re.findall(number, text)
for i in text:
    if i.lower() not in char and i != ' ':
        text = text.replace(i, '')

print("Lọc ra số :",findnum)
print("Lọc ra text :",text)








