import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Load dataset
def load_data(base_dir, metadata_path):
    """Load metadata and image paths."""
    print("[INFO] Loading metadata...")
    metadata = pd.read_csv(metadata_path)

    image_paths = []
    for image_id in metadata['image_id']:
        part1_path = os.path.join(base_dir, 'HAM10000_images_part_1', f'{image_id}.jpg')
        part2_path = os.path.join(base_dir, 'HAM10000_images_part_2', f'{image_id}.jpg')

        if os.path.exists(part1_path):
            image_paths.append(part1_path)
        elif os.path.exists(part2_path):
            image_paths.append(part2_path)
        else:
            raise FileNotFoundError(f"Image {image_id}.jpg not found.")

    metadata['image_path'] = image_paths
    metadata['label'] = metadata['dx'].astype('category').cat.codes
    print("[INFO] Metadata loaded successfully.")
    return metadata

# Preprocess images
def preprocess_data(metadata):
    """Preprocess image data and return numpy arrays."""
    print("[INFO] Preprocessing images...")
    data, labels = [], []
    for _, row in metadata.iterrows():
        try:
            img = Image.open(row['image_path']).convert('RGB')
            img_resized = img.resize((64, 64))
            data.append(np.array(img_resized))
            labels.append(row['label'])
        except Exception as e:
            print(f"[WARNING] Skipping image {row['image_path']} due to error: {e}")

    print("[INFO] Preprocessing complete.")
    return np.array(data, dtype='float32') / 255.0, np.array(labels)

# Build CNN model
def build_model():
    """Build the CNN model."""
    print("[INFO] Building CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 output classes
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("[INFO] Model built successfully.")
    return model

# Main function
def main():
    """Main function to run the pipeline."""
    base_dir = './datasets'
    metadata_path = os.path.join(base_dir, 'HAM10000_metadata.csv')

    # Load and preprocess data
    metadata = load_data(base_dir, metadata_path)
    data, labels = preprocess_data(metadata)

    # Split dataset
    print("[INFO] Splitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    # Build model
    model = build_model()

    # Data augmentation
    print("[INFO] Setting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train model
    print("[INFO] Training model...")
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                        validation_data=(x_test, y_test),
                        epochs=50,
                        callbacks=[early_stopping, reduce_lr])

    # Save model
    model.save('skin_cancer_model.h5')
    print("[INFO] Model saved as 'skin_cancer_model.h5'.")

    # Evaluate model
    print("[INFO] Evaluating model...")
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot training history
    print("[INFO] Plotting training history...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

if __name__ == "__main__":
    main()