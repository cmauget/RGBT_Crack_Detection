import streamlit as st #type: ignore
from utils import Image_Utils
import cv2 #type: ignore
import torch#type: ignore
from model import CNNModel
from dataset import CustomDataset
from PIL import Image#type: ignore
from torch.utils.data import DataLoader #type:ignore
import numpy as np#type:ignore
from torchvision import transforms#type:ignore
from tqdm import tqdm#type:ignore

CH = 4

def predict_and_draw_boxes(image, model, device, uploaded_file):
    """Predict cracks on an image and draw green boxes around them

    Receives an image, split it in 96px by 96px square.
    The images are passed thriugh the given model and returns the inputed image the image with green boxes if a true prediction

    Parameters
    ----
    image: numpy.ndarray
        The image to predict
    model: torch.nn.Module
        The model to use for prediction

    Returns
    ----
    image: numpy.ndarray
       The image with green boxes around the cracks
    """

    image2 = Image_Utils.load_streamlit(uploaded_file)
    (b, g, r, _) = cv2.split(image2)
    image2 = cv2.merge((b, g, r))
    
    image_width, image_height, _ = image2.shape

    square_size = 96
    stride = 48

    transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

    num_rows = (image_width - square_size) // stride + 1
    num_cols = (image_height - square_size) // stride + 1

    for row in tqdm(range(num_rows)):
        for col in range(num_cols):
            left = col * stride
            top = row * stride
            right = left + square_size
            bottom = top + square_size

            square = image.crop((left, top, right, bottom))
            square = transform(square)
            square = square.to(device)

            with torch.no_grad():
                model.eval()
                outputs = model(square.unsqueeze(0))
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu().numpy()

                if predicted[0] == 1:
                    overlay = image2.copy()
                    col = (0, 255, 0)  
                    alpha = 1

                    cv2.rectangle(overlay,(left, top), (right, bottom),  col, 2) 
    
                    image2 = cv2.addWeighted(overlay, alpha, image2, 1 - alpha, 0)
    return image2


def main():

    ch = CH
    st.title("Classifier")
    st.write("Upload the image")
    uploaded_file = st.sidebar.file_uploader("Import an Image", type=["png", "jpg", "jpeg","tif"])
    checkbox_value = st.sidebar.checkbox("3 channel prediction")

    if checkbox_value:
        ch=3
    
    if uploaded_file is not None:
        image = Image_Utils.load_streamlit_PIL(uploaded_file)
        r, g, b, _ = image.split()
        image_rgb = Image.merge("RGB", (r, g, b))
        if ch==3:
            image = image_rgb
        else: pass
        
        if st.button("Detect cracks"): 
            device = "mps"
            model = CNNModel(device_=device, ch=ch)
            if ch==3:
                model.load_state_dict(torch.load('models/RGB.pth'))
            else:
                model.load_state_dict(torch.load('models/RGBT.pth'))
            
            with st.spinner('Wait for it...'):
                image = predict_and_draw_boxes(image, model, device, uploaded_file)

            st.image(image, use_column_width=True, caption="Image with cracks detected")
            Image_Utils.save(f"output/output_crack_{ch}ch.png", image)

    
if __name__ == "__main__":
    main()

    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)