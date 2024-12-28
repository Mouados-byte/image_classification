import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Create directories for saving plots and model
for directory in ['plots', 'model']:
    if not os.path.exists(directory):
        os.makedirs(directory)

MODEL_PATH = 'model/fashion_mnist_mlp.h5'

def load_and_preprocess_data():
    # Load and preprocess the Fashion-MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten the images
    x_train = x_train.reshape((-1, 28*28))
    x_test = x_test.reshape((-1, 28*28))
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def train_or_load_model():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        # Quick evaluation to verify the model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f'Loaded model test accuracy: {test_accuracy:.3f}')
        return model, None
    
    print("Training new model...")
    model = create_model()
    model.summary()
    
    # Train the model
    history = model.fit(x_train, y_train,
                       batch_size=32,
                       epochs=10,
                       validation_split=0.2,
                       verbose=1)
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy: {test_accuracy:.3f}')
    
    return model, history

def save_training_history(history):
    if history is None:
        print("No training history available (loaded from cached model)")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['Train', 'Validation'])
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.close()
    print("Training history plot saved as 'plots/training_history.png'")

def save_confusion_matrix(model):
    # Get test data
    _, (x_test, y_test) = load_and_preprocess_data()
    
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'plots/confusion_matrix.png'")

def save_predictions(model):
    # Get test data
    _, (x_test, y_test) = load_and_preprocess_data()
    
    # Get random test images
    num_images = 5
    test_indices = np.random.randint(0, x_test.shape[0], num_images)
    
    # Make predictions
    test_images = x_test[test_indices]
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test[test_indices]
    
    # Display images and predictions
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        plt.title(f'Pred: {class_names[predicted_classes[i]]}\nTrue: {class_names[true_classes[i]]}',
                 color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/predictions.png')
    plt.close()
    print("Prediction examples saved as 'plots/predictions.png'")

def main():
    # Train or load model
    model, history = train_or_load_model()
    
    # Save all plots
    save_training_history(history)
    save_confusion_matrix(model)
    save_predictions(model)
    
    print("\nAll plots have been saved in the 'plots' directory.")
    print("1. Training history: plots/training_history.png (if newly trained)")
    print("2. Confusion matrix: plots/confusion_matrix.png")
    print("3. Prediction examples: plots/predictions.png")

if __name__ == "__main__":
    main()