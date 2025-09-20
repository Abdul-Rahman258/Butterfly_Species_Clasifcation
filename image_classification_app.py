import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import streamlit as st
from io import BytesIO
import threading
import os

# FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and label map
model = tf.keras.models.load_model("Butterfly_species_classification_model.h5")

# Load label map from CSV
def load_label_map(csv_path):
    df = pd.read_csv(csv_path)
    unique_labels = sorted(df['label'].unique())  # Adjust 'label' if column name differs
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    return reverse_label_map

# Try loading from cleaned_dataset.csv or Training_set.csv
csv_path = "cleaned_dataset.csv" if os.path.exists("cleaned_dataset.csv") else "Training_set.csv"
reverse_label_map = load_label_map(csv_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = reverse_label_map[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

# Streamlit App
def run_streamlit():
    st.title("Butterfly Species Classfication By Abdul Rahman Baig")
    st.markdown("Upload an image to get a classification prediction.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Making prediction..."):
                img = Image.open(uploaded_file)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = reverse_label_map[predicted_class_idx]
                confidence = float(predictions[0][predicted_class_idx])
                
                st.success(f"**Prediction**: {predicted_class}")
                st.info(f"**Confidence**: {confidence:.2%}")

# Main execution
if __name__ == "__main__":
    # Run FastAPI server in background
    api_thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000))
    api_thread.daemon = True
    api_thread.start()
    
    # Run Streamlit app
    run_streamlit()