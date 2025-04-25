import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Paths
base_dir = os.path.join(os.getcwd(), 'AorT')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
img_height, img_width = 128, 128
batch_size = 32
val_split = 0.2

# Feature Extractor
def get_features_and_labels(directory, datagen, label_map):
    generator = datagen.flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    features = base_model.predict(generator, verbose=1)
    labels = generator.classes
    return features.reshape(features.shape[0], -1), labels

# Load pre-trained VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Image preprocessing
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_X, train_y = get_features_and_labels(train_dir, datagen, None)
test_X, test_y = get_features_and_labels(test_dir, datagen, None)

# Class names
class_names = os.listdir(train_dir)

# Models
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier()
}

accuracies = {}

# Plotting function
def plot_confusion_matrix(cm, name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Train & evaluate each model
for name, clf in models.items():
    print(f"\n🔧 Training {name}...")
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    acc = accuracy_score(test_y, preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(test_y, preds, target_names=class_names))
    cm = confusion_matrix(test_y, preds)
    plot_confusion_matrix(cm, name)

# Comparison bar graph
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='teal')
plt.title('🔍 Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()
