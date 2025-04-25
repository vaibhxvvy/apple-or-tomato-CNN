# This Python script is a machine learning project that involves training a model to classify images
# using transfer learning with the VGG16 model. Here is a breakdown of what the script does:
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# Set directories
base_dir = os.path.join(os.getcwd(), 'AorT')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_split = 0.2
cache_dir = 'cache_aort'

# Cache file paths
train_X_path = os.path.join(cache_dir, 'train_X.npy')
train_y_path = os.path.join(cache_dir, 'train_y.npy')
val_X_path = os.path.join(cache_dir, 'val_X.npy')
val_y_path = os.path.join(cache_dir, 'val_y.npy')
test_X_path = os.path.join(cache_dir, 'test_X.npy')
test_y_path = os.path.join(cache_dir, 'test_y.npy')
class_labels_path = os.path.join(cache_dir, 'class_names.npy')

# Image parameters
img_height, img_width = 128, 128
batch_size = 32
epochs = 30  # Increased epochs

# Load from cache if exists
if all(os.path.exists(p) for p in [
    train_X_path, train_y_path,
    val_X_path, val_y_path,
    test_X_path, test_y_path,
    class_labels_path
]):
    print("🔁 Loading data from cache...")

    train_X = np.load(train_X_path)
    train_y = np.load(train_y_path)
    val_X = np.load(val_X_path)
    val_y = np.load(val_y_path)
    test_X = np.load(test_X_path)
    test_y = np.load(test_y_path)
    class_names = np.load(class_labels_path)
else:
    print("📦 Caching preprocessed data for first time...")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=val_split
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    class_names = np.array(list(train_gen.class_indices.keys()))

    def extract_data(generator):
        X, y = [], []
        for data, labels in generator:
            X.append(data)
            y.append(labels)
            if len(X) * batch_size >= generator.samples:
                break
        return np.concatenate(X), np.concatenate(y)

    train_X, train_y = extract_data(train_gen)
    val_X, val_y = extract_data(val_gen)
    test_X, test_y = extract_data(test_gen)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(train_X_path, train_X)
    np.save(train_y_path, train_y)
    np.save(val_X_path, val_X)
    np.save(val_y_path, val_y)
    np.save(test_X_path, test_X)
    np.save(test_y_path, test_y)
    np.save(class_labels_path, class_names)

# Build the VGG16-based model with more fine-tuning
base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
base_model.trainable = True  # Unfreeze all layers

# Fine-tune more layers of VGG16
for layer in base_model.layers[:15]:  # Unfreeze layers after the 15th layer
    layer.trainable = False

# Define the model with more layers and dropout for regularization
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),  # Added dropout
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Added dropout
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=epochs, batch_size=batch_size)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Evaluate the model
preds = model.predict(test_X)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(test_y, axis=1)

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
print("🧩 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
