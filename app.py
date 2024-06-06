import streamlit as st
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

model_name = 'DeepLabV3-resnet50'

class_dict = {
    (0, 255, 255): 0,  # Urban Land
    (255, 255, 0): 1,  # Agriculture Land
    (255, 0, 255): 2,  # Rangeland
    (0, 255, 0): 3,    # Forest Land
    (0, 0, 255): 4,    # Water
    (255, 255, 255): 5,  # Barren Land
    (0, 0, 0): 6      # Unknown
}

class_list = ["Urban Land", "Agriculture Land", "Rangeland", "Forest Land", "Water", "Barren Land", "Unknown"]

# Function to convert one-hot encoded image to RGB image
def one_hot_rgb(one_hot_img, classDict):
    height, width = one_hot_img.shape
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)

    for color, category in classDict.items():
        category_mask = (one_hot_img == category)
        color = np.array(color)

        rgb_img[category_mask] = color

    return rgb_img

# Load the model
model = torch.load(model_name + '.pth')
with torch.no_grad():
    model.eval()

image_transforms = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor()
])

# Prediction function
def predict(image):
    image = image_transforms(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(image)
    output = logits.sigmoid()
    return output

# Function to convert tensor to image
def tensor_to_rgb_mask(tensor, class_dict):
    tensor = tensor.squeeze().cpu().numpy()
    pr_mask = np.argmax(tensor, axis=0)
    rgb_mask = one_hot_rgb(pr_mask, class_dict)
    return rgb_mask

# Function to create a legend image
def create_legend(class_dict, class_list):
    legend_img = np.zeros((50 * len(class_list), 300, 3), dtype=np.uint8)
    for i, (color, class_name) in enumerate(zip(class_dict.keys(), class_list)):
        legend_img[i*50:(i+1)*50, :50] = color
        cv2.putText(legend_img, class_name, (60, (i*50)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return legend_img

# Streamlit app
st.title('D33P Learning on Edge: Land Cover Classification')
st.markdown('Authors: Sebastián González, Andreína Sanánez, Luis Javier Karam')
st.write('This is a demo of a land cover classification model using PyTorch and DeepLabV3 with a ResNet-50 backbone.')
st.markdown('You can download sample images from [this Google Drive](https://drive.google.com/drive/folders/1iAFViqL08HmDWSYXsu-1SUwS9dWy1-ct?usp=sharing).')
st.toast('Welcome to our land cover classification model demo!', icon='🌍')

uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating mask...")
    predictions = predict(image)
    
    mask = tensor_to_rgb_mask(predictions, class_dict)
    st.image(mask, caption='Generated Mask', use_column_width=True)

     # Create and display legend
    legend_img = create_legend(class_dict, class_list)
    st.image(legend_img, caption='Classes', use_column_width=False)