import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import streamlit as st
from io import BytesIO

# Data Cleaning
def clean_dataset(input_folder, csv_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_path)
    valid_images = []
    valid_labels = []
    
    for _, row in df.iterrows():
        image_name = row['filename']  # Adjust if column name differs
        label = row['label']  # Adjust if column name differs
        image_path = os.path.join(input_folder, image_name)
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Invalid image: {image_name}")
                continue
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, img)
            valid_images.append(image_name)
            valid_labels.append(label)
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue
    
    cleaned_df = pd.DataFrame({
        'filename': valid_images,
        'label': valid_labels
    })
    cleaned_df.to_csv(os.path.join(output_folder, 'cleaned_dataset.csv'), index=False)
    
    return valid_images, valid_labels, cleaned_df

# CNN Model
def create_cnn_model(num_classes, input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Data Generator
def create_data_generator(df, directory, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=directory,
        x_col='filename',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )
    
    val_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=directory,
        x_col='filename',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=True
    )
    
    return train_generator, val_generator, train_generator.class_indices

# FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and label map
model = None
reverse_label_map = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, reverse_label_map
    
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
    st.title("Image Classification App")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict"):
            img = Image.open(uploaded_file)
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = reverse_label_map[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2%}")

# Main execution
if __name__ == "__main__":
    # Clean dataset
    input_folder = "train"
    csv_path = "Training_set.csv"
    output_folder = "cleaned_train"
    
    image_list, label_list, cleaned_df = clean_dataset(input_folder, csv_path, output_folder)
    
    # Create data generators
    train_generator, val_generator, label_map = create_data_generator(cleaned_df, output_folder)
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Create and train model
    model = create_cnn_model(num_classes=len(label_map))
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size
    )
    
    # Save model
    model.save("image_classification_model.h5")
    
    # Run FastAPI server in background
    import threading
    api_thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000))
    api_thread.daemon = True
    api_thread.start()
    
    # Run Streamlit app
    run_streamlit()