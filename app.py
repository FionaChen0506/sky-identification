import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import gradio as gr

def read_image(image_path):
    # Convert to string if it's not already
    image_path = str(image_path)
    
    # Read an image
    image = cv.imread(image_path)
    
    return image

def convert_to_rgb(image):
    # Convert BGR to RGB
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def convert_to_grayscale(image):
    # Convert to grayscale
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def apply_adaptive_threshold(gray_image):
    # Apply adaptive thresholding
    return cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


def apply_canny_edge_detection(gray_image):
    # Apply Canny Edge Detection
    return cv.Canny(gray_image, 100, 150)

def dilate_edges(edges):
    # Dilate the edges to emphasize larger features
    return cv.dilate(edges, None, iterations=2)

def invert_edges(edges):
    # Invert the edges
    return cv.bitwise_not(edges)

def combine_masks(adaptive_threshold, inverted_edges):
    # Combine the masks
    return cv.bitwise_and(adaptive_threshold, inverted_edges)

def apply_morphological_operations(combined_mask):
    # Morphological operations
    morpho_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (14, 14))
    return cv.morphologyEx(combined_mask, cv.MORPH_OPEN, morpho_kernel)

def post_process_mask(mask_morpho):
    # Additional morphological operations for post-processing
    morpho_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (14, 14))
    return cv.morphologyEx(mask_morpho, cv.MORPH_CLOSE, morpho_kernel)

def make_sky_white(image, post_processed_mask):
    # Create a new image with the same shape and size as the original one
    white_sky = np.zeros(image.shape, dtype=np.uint8)
    
    # Make the sky white
    white_sky[post_processed_mask == 255] = [255, 255, 255]
    
    return white_sky

def merge_images(image, white_sky):
    # Merge two images
    result_image_bgr = cv.add(image, white_sky)
    return cv.cvtColor(result_image_bgr, cv.COLOR_BGR2RGB)

def process_image(image_path):
    # Read an image
    image = read_image(image_path)
    if image is None:
        return None
    
    # Convert BGR to RGB
    image_rgb = convert_to_rgb(image)
    
    # Convert to grayscale
    gray_image = convert_to_grayscale(image)
    
    # Apply adaptive thresholding
    adaptive_threshold = apply_adaptive_threshold(gray_image)
    
    # Apply Canny Edge Detection
    edges = apply_canny_edge_detection(gray_image)
    
    # Dilate edges
    dilated_edges = dilate_edges(edges)
    
    # Invert edges
    inverted_dilated_edges = invert_edges(dilated_edges)
    
    # Combine masks
    combined_mask = combine_masks(adaptive_threshold, inverted_dilated_edges)
    
    # Morphological operations
    mask_morpho = apply_morphological_operations(combined_mask)
    
    # Post-processing
    post_processed_mask = post_process_mask(mask_morpho)
    
    # Make sky white
    white_sky = make_sky_white(image, post_processed_mask)
    
    # Merge images
    result_image = merge_images(image, white_sky)
    
    return result_image

# Create a Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs= gr.Image(type='filepath'),
    outputs='image',
    live=True,
    title='Sky Identification',
    examples=["./pics/pic1.jpg", "./pics/pic12.jpg", "./pics/pic9.jpg", "./pics/pic18.jpg"]
)

# Launch the interface
iface.launch()

