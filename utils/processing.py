# utils/processing.py
import cv2
import numpy as np

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_binary(gray_img, threshold=127):
    _, binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def blur_image(img_np):
    return cv2.GaussianBlur(img_np, (5, 5), 0)

def apply_morphology(img_bin, operation="dilation", se_shape="rect"):
    if se_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    elif se_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = np.ones((5, 5), np.uint8)

    if operation == "dilation":
        return cv2.dilate(img_bin, kernel, iterations=1)
    else:
        return cv2.erode(img_bin, kernel, iterations=1)

# Operasi Aritmatika
def arithmetic_add(img1, img2):
    return cv2.add(img1, img2)

def arithmetic_subtract(img1, img2):
    return cv2.subtract(img1, img2)

def arithmetic_multiply(img1, img2):
    return cv2.multiply(img1, img2)

def arithmetic_divide(img1, img2):
    # Hindari pembagian dengan nol
    img2 = np.where(img2 == 0, 1, img2)
    return cv2.divide(img1, img2)

# Operasi Logika
def logic_and(img1, img2):
    return cv2.bitwise_and(img1, img2)

def logic_or(img1, img2):
    return cv2.bitwise_or(img1, img2)

def logic_xor(img1, img2):
    return cv2.bitwise_xor(img1, img2)

def logic_not(img1):
    return cv2.bitwise_not(img1)
