import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def evaluate_model(model_path, test_data_dir):
    # Load the model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Lists to store true and predicted labels
    y_true = []
    y_pred = []
    predictions = []
    
    # Class labels
    class_labels = ["Mild", "Moderate", "Severe", "Normal"]
    
    print("\nProcessing images...")
    # Process each class folder
    for class_idx, class_name in enumerate(class_labels):
        class_dir = os.path.join(test_data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found for class {class_name}")
            continue
            
        print(f"\nProcessing {class_name} images...")
        # Process each image in the class folder
        for image_name in os.listdir(class_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(class_dir, image_name)
            print(f"Processing {image_name}...")
            
            # Preprocess and predict
            processed_image = preprocess_image(image_path)
            prediction = model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Store true and predicted labels
            y_true.append(class_idx)
            y_pred.append(predicted_class)
            predictions.append({
                'file': image_name,
                'true': class_name,
                'predicted': class_labels[predicted_class],
                'confidence': confidence
            })
    
    # Print individual predictions
    print("\nIndividual Predictions:")
    for pred in predictions:
        print(f"File: {pred['file']}")
        print(f"True: {pred['true']}, Predicted: {pred['predicted']}")
        print(f"Confidence: {pred['confidence']:.2%}\n")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Get model path and test directory from user
    model_path = input("Enter the path to keras_model.h5 (or press Enter for default 'keras_model.h5'): ").strip() or 'keras_model.h5'
    test_data_dir = input("Enter the path to test data directory (or press Enter for default 'test_data'): ").strip() or 'test_data'
    
    print("\nStarting model evaluation...")
    print(f"Model path: {model_path}")
    print(f"Test data directory: {test_data_dir}")
    
    evaluate_model(model_path, test_data_dir)
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'") 