import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import google.generativeai as genai

# Configure working directory and model paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Configure the Gemini API key
GOOGLE_API_KEY = "AIzaSyCD_EwxxvFhQXj9o_yvShWxUGx6J-uvOIM"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini 2.0 Flash-Lite model
model_name = "models/gemini-2.0-flash-lite"

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model(model_path)

# Load the class indices
with open(f"{working_dir}/class_indices.json") as f:
    class_indices = json.load(f)

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Scale to [0, 1]
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to get disease information from Gemini using GenerativeModel
def get_disease_info(predicted_class_name):
    prompt = f"""
    Please provide a detailed explanation in Bengali for the plant disease **{predicted_class_name}**. The response should be simple, concise, and easy for anyone to understand. Use markdown formatting with headings and bullet points, and include relevant emojis to make it visually engaging. The explanation must cover the following points:

    1. **রোগের নাম ও প্রভাবিত গাছপালা** 🌱  
    - (Explain the disease name and which plants are affected)

    2. **রোগের কারণ** 🔍  
    - (Describe the pathogen type and scientific name, if available)

    3. **লক্ষণাবলী** ⚠️  
    - (List the main symptoms observed on the plants)

    4. **ফসলের উপর প্রভাব** 📉  
    - (Discuss how the disease might impact crop yield)

    5. **প্রতিরোধ ও চিকিৎসা** 💊  
    - (Recommend treatments and preventive measures)

    Make sure the output is structured with markdown (using headings, bullet points, and emojis) and is written entirely in Bengali.

    """
    try:
        gemini_model = genai.GenerativeModel(model_name)
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating disease information: {str(e)}\n\nPlease check your API key and model name."

# Streamlit App
st.title('🌱 Plant Disease Identification and Information System')
st.write("Upload an image of a diseased plant leaf to identify the disease and get detailed information.")

uploaded_image = st.file_uploader("Upload an image of the plant leaf...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # col1, col2 = st.columns(2)
    # with col1:
    st.subheader("Uploaded Image")
    resized_img = image.resize((250, 250))
    st.image(resized_img)

    # with col2:
    st.subheader("Disease Identification")
    if st.button('Identify Disease'):
        with st.spinner('Analyzing the image...'):
            prediction = predict_image_class(model, image, class_indices)
            st.success(f'Detected Disease: {prediction}')

        with st.spinner('Gathering information about the disease...'):
            disease_info = get_disease_info(prediction)
        st.subheader("Disease Information")
        st.markdown(disease_info)

        # Additional options for specific questions
        st.subheader("Additional Questions")
        user_query = st.text_input("Ask a specific question about this disease:")

        if user_query and st.button('Get Answer'):
            specific_prompt = f"Question about {prediction} plant disease: {user_query}"
            with st.spinner('Generating response...'):
                try:
                    gemini_model = genai.GenerativeModel(model_name)
                    specific_response = gemini_model.generate_content(specific_prompt)
                    st.markdown("**Answer:**")
                    st.markdown(specific_response.text)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

# Footer
st.markdown("---")
st.markdown(
    "This application combines computer vision for disease detection and generative AI for providing comprehensive information about plant diseases."
)
