import streamlit as st
import PIL
import tf_keras
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import os

# Load label map
labels_file = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels_file)
labels = dict(zip(df.id, df.name))

# Define model URL
model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'

# Load the model
classifier = tf_keras.Sequential([
    hub.KerasLayer(model_url, input_shape=(321, 321, 3), output_key="predictions:logits")
])

def image_processing(image):
    img_shape = (321, 321)
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    result = classifier.predict(img_array)
    return labels[np.argmax(result)], img

def get_map(location_name):
    geolocator = Nominatim(user_agent="Landmark_Recognition_App")
    location = geolocator.geocode(location_name)
    if location:
        return location.address, location.latitude, location.longitude
    else:
        return None, None, None

def run():
    st.title("🌏 Landmark Recognition")
    
    # Display logo
    try:
        img = PIL.Image.open('logo.png')
        img = img.resize((256, 256))
        st.image(img)
    except FileNotFoundError:
        st.warning("Logo image not found. Please ensure 'logo.png' is in the project directory.")
    
    # File uploader for image input
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg', 'jpeg'])
    
    if img_file is not None:
        # Save the uploaded image
        save_image_path = os.path.join('./Uploaded_Images/', img_file.name)
        os.makedirs('./Uploaded_Images/', exist_ok=True)  # Create directory if it doesn't exist
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        st.write("Image saved to:", save_image_path)  # Debug information

        # Image processing and prediction
        prediction, processed_image = image_processing(save_image_path)
        st.image(processed_image, caption='Uploaded Image', use_column_width=True)
        st.header(f"📍 **Predicted Landmark: {prediction}**")
        
        # Attempt to get location information
        address, latitude, longitude = get_map(prediction)
        st.write("Address, Latitude, Longitude retrieved:", address, latitude, longitude)  # Debug information
        if address:
            st.success(f'Address: {address}')
            loc_dict = {'Latitude': latitude, 'Longitude': longitude}
            st.subheader(f'✅ **Coordinates for {prediction}**')
            st.json(loc_dict)

            # Display map
            data = [[latitude, longitude]]
            df = pd.DataFrame(data, columns=['lat', 'lon'])
            st.subheader(f'✅ **{prediction} on the Map** 🗺️')
            st.map(df)
        else:
            st.warning("Address not found for the predicted landmark.")

            
if __name__ == "__main__":
    run()
