import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, classification_report
import tensorflow as tf

# Add custom object scope to handle DepthwiseConv2D compatibility
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

def evaluate_model(model_path, test_data_dir):
    # Load the model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Class labels
    class_labels = ["Mild", "Moderate", "Severe", "Normal"]
    
    # Lists to store true and predicted labels
    y_true = []
    y_pred = []
    
    # Dictionary to store results
    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': defaultdict(int)})
    
    print("\nProcessing images...")
    # Process each class folder
    for class_idx, true_class in enumerate(class_labels):
        class_dir = os.path.join(test_data_dir, true_class)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found for class {true_class}")
            continue
            
        print(f"\nProcessing {true_class} images...")
        # Count total images in this class
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        results[true_class]['total'] = len(image_files)
        
        # Process each image
        for image_name in image_files:
            image_path = os.path.join(class_dir, image_name)
            print(f"Processing {image_name}...")
            
            # Preprocess and predict
            processed_image = preprocess_image(image_path)
            prediction = model.predict(processed_image, verbose=0)
            predicted_idx = np.argmax(prediction[0])
            predicted_class = class_labels[predicted_idx]
            confidence = prediction[0][predicted_idx]
            
            # Store true and predicted labels
            y_true.append(class_idx)
            y_pred.append(predicted_idx)
            results[true_class]['predictions'][predicted_class] += 1
            
            print(f"  Predicted as {predicted_class} with confidence: {confidence:.2%}")
    
    # Calculate metrics using scikit-learn
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_labels)))
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 50)
    for i, class_name in enumerate(class_labels):
        print(f"\n{class_name} Class:")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall: {recall[i]:.4f}")
        print(f"F1-Score: {f1[i]:.4f}")
        print(f"Support: {support[i]}")
    
    # Calculate and display confusion matrix
    print("\nPrediction Distribution:")
    print("-" * 50)
    for class_name in class_labels:
        class_total = results[class_name]['total']
        if class_total == 0:
            continue
            
        print(f"\nFor {class_name} images:")
        for predicted, count in results[class_name]['predictions'].items():
            percentage = (count / class_total) * 100
            print(f"  Predicted as {predicted}: {count} ({percentage:.2f}%)")
    
    # Create a simple bar plot of class distribution
    plt.figure(figsize=(10, 6))
    class_counts = [results[label]['total'] for label in class_labels]
    plt.bar(class_labels, class_counts)
    plt.title('Distribution of Images Across Classes')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default paths in the temp directory
    default_model = os.path.join(current_dir, 'keras_model.h5')
    default_test_dir = os.path.join(current_dir, 'test')
    
    print("\nChecking paths:")
    print(f"Current directory: {current_dir}")
    print("\nAvailable files in current directory:")
    for file in os.listdir(current_dir):
        print(f"  - {file}")
    
    # Use the correct paths by default
    model_path = default_model
    test_data_dir = default_test_dir
    
    # Verify paths exist
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        exit(1)
    
    if not os.path.exists(test_data_dir):
        print(f"\nERROR: Test directory not found at {test_data_dir}")
        exit(1)
    
    print("\nStarting evaluation...")
    print(f"Model path: {model_path}")
    print(f"Test data directory: {test_data_dir}")
    
    evaluate_model(model_path, test_data_dir)
    print("\nClass distribution plot has been saved as 'class_distribution.png'") 