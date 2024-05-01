# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 07:27:23 2024

@author: delll
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu


def apply_gaussian_blur(image, kernel_size):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def apply_canny_edge_detection(image, min_threshold, max_threshold):
    edges = cv2.Canny(image, min_threshold, max_threshold)
    return edges

def adjust_brightness_contrast(image, brightness, contrast):
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted_image

def apply_image_segmentation(image, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, segmented_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return segmented_image

def apply_histogram_equalization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

def apply_image_restoration(image, method='denoise'):
    if method == 'denoise':
        restored_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    elif method == 'deblur':
        restored_image = cv2.medianBlur(image, 5)
    else:
        restored_image = image
    return restored_image

def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

def convert_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def convert_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

def plot_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 5))
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    st.pyplot()
    
def apply_laplacian_of_gaussian(image, kernel_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian_image = np.uint8(np.absolute(laplacian))
    return cv2.cvtColor(laplacian_image, cv2.COLOR_GRAY2RGB)

def apply_difference_of_gaussians(image, k1, k2):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray_image, (k1, k1), 0)
    blurred2 = cv2.GaussianBlur(gray_image, (k2, k2), 0)
    dog_image = blurred1 - blurred2
    return cv2.cvtColor(dog_image, cv2.COLOR_GRAY2RGB)

def main():
    st.title(" 🖥️ Image Processing App")

    uploaded_file = st.file_uploader("Upload an image🖼️ ", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
    
        with st.sidebar:
            st.subheader("Select an Image Processing Task")
            # Image processing options
            option = st.selectbox("Select an option", ["None",'Color Space Conversion',"Gaussian Blur", "Edge Detection Variants", "Adjust Brightness/Contrast","Image Segmentation","Histogram Equalization", "Image Restoration"])
    
            if option == "Gaussian Blur":
                kernel_size = st.slider("Select kernel size", 3, 31, 3, step=2)
                processed_image = apply_gaussian_blur(image, kernel_size)
    
            elif option == "Edge Detection Variants":
                Edge_Detection_method = st.radio("Select Edge Detection method", ["Canny Edge Detection","Laplacian of Gaussian (LoG)","Difference of Gaussians (DoG)"])
                if Edge_Detection_method=="Canny Edge Detection":
                    min_threshold = st.slider("Select min threshold", 0, 255, 50)
                    max_threshold = st.slider("Select max threshold", 0, 255, 150)
                    processed_image = apply_canny_edge_detection(image, min_threshold, max_threshold)
                elif Edge_Detection_method == "Laplacian of Gaussian (LoG)":
                    kernel_size = st.slider("Select kernel size", 3, 31, 3, step=2)
                    processed_image = apply_laplacian_of_gaussian(image, kernel_size)
    
                elif Edge_Detection_method == "Difference of Gaussians (DoG)":
                    k1 = st.slider("Select kernel size 1", 3, 31, 3, step=2)
                    k2 = st.slider("Select kernel size 2 (greater than kernel size 1)", k1 + 1, 33, 5, step=2)
                    processed_image = apply_difference_of_gaussians(image, k1, k2)
        
    
            elif option == "Adjust Brightness/Contrast":
                brightness = st.slider("Brightness", -100, 100, 0)
                contrast = st.slider("Contrast", 0.0, 3.0, 1.0)
                processed_image = adjust_brightness_contrast(image, brightness, contrast)
                
            elif option == "Image Segmentation":
                threshold = st.slider("Select segmentation threshold", 0, 255, 128)
                processed_image = apply_image_segmentation(image, threshold)
            
            elif option == "Histogram Equalization":
                st.set_option('deprecation.showPyplotGlobalUse', False)
                processed_image = apply_histogram_equalization(image)
                plot_histogram(image)
    
            elif option == "Image Restoration":
                restoration_method = st.radio("Select restoration method", ["Denoise", "Deblur"])
                processed_image = apply_image_restoration(image, method=restoration_method.lower())
            
            elif option == "Color Space Conversion":
                color_space_method = st.radio("Select color space conversion method", ["Grayscale","HSV","LAB"])
                if color_space_method=="Grayscale":
                    processed_image = convert_to_grayscale(image)
                
                if color_space_method=="HSV":
                    processed_image = convert_to_hsv(image)
                
                if color_space_method=="LAB":
                    processed_image = convert_to_lab(image)
    
            else:
                processed_image = image_rgb

        st.image(processed_image, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
