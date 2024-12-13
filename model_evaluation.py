import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Custom layer definition (same as in app.py)
@tf.keras.utils.register_keras_serializable()
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        if 'groups' in config:
            del config['groups']
        return config

# Register the custom layer
tf.keras.utils.get_custom_objects().update({
    'DepthwiseConv2D': CustomDepthwiseConv2D
})

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def evaluate_model(test_data_dir):
    # Load the model
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    
    # Lists to store true and predicted labels
    y_true = []
    y_pred = []
    
    # Class labels
    class_labels = ["Mild", "Moderate", "Severe", "Normal"]
    
    # Process each class folder
    for class_idx, class_name in enumerate(class_labels):
        class_dir = os.path.join(test_data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found for class {class_name}")
            continue
            
        # Process each image in the class folder
        for image_name in os.listdir(class_dir):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(class_dir, image_name)
            
            # Preprocess and predict
            processed_image = preprocess_image(image_path)
            prediction = model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(prediction[0])
            
            # Store true and predicted labels
            y_true.append(class_idx)
            y_pred.append(predicted_class)
    
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
    # Specify your test data directory
    # The directory structure should be:
    # test_data/
    #   ├── Mild/
    #   ├── Moderate/
    #   ├── Severe/
    #   └── Normal/
    
    test_data_dir = "test_data"  # Change this to your test data directory
    
    print("Starting model evaluation...")
    evaluate_model(test_data_dir)
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'") 