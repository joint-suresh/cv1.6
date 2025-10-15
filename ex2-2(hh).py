'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os, zipfile, shutil

url = "https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip"
path = tf.keras.utils.get_file('horse-or-human.zip', origin=url)
extract_dir = os.path.join(os.path.dirname(path), 'horse-or-human')
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

horse_dir = os.path.join(extract_dir, 'horses')
human_dir = os.path.join(extract_dir, 'humans')
os.makedirs(horse_dir, exist_ok=True)
os.makedirs(human_dir, exist_ok=True)

for file in os.listdir(extract_dir):
    file_path = os.path.join(extract_dir, file)
    if os.path.isfile(file_path):
        if file.startswith('horse'):
            shutil.move(file_path, os.path.join(horse_dir, file))
        elif file.startswith('human'):
            shutil.move(file_path, os.path.join(human_dir, file))

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(
    directory=extract_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_data = datagen.flow_from_directory(
    directory=extract_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

x_batch, y_batch = next(train_data)
plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_batch[i])
    plt.title("Human" if y_batch[i] == 1 else "Horse")
    plt.axis('off')
plt.show()

base_model = ResNet50(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=val_data, epochs=5)

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

'''





import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os, shutil

# Absolute paths to the horses and humans directories
horse_dir = "C:/Users/SRIRAM/.keras/datasets/horse-or-human/horses"
human_dir = "C:/Users/SRIRAM/.keras/datasets/horse-or-human/humans"

# Ensure that directories exist (you mentioned the directories already exist, so this step is optional)
os.makedirs(horse_dir, exist_ok=True)
os.makedirs(human_dir, exist_ok=True)

# Using ImageDataGenerator to load images from the absolute directories
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Use the absolute paths for loading data
train_data = datagen.flow_from_directory(
    directory="C:/Users/SRIRAM/.keras/datasets/horse-or-human",  # Root directory containing 'horses' and 'humans'
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Training data subset
)

val_data = datagen.flow_from_directory(
    directory="C:/Users/SRIRAM/.keras/datasets/horse-or-human",  # Root directory containing 'horses' and 'humans'
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Validation data subset
)

# Display some images to check
x_batch, y_batch = next(train_data)
plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_batch[i])
    plt.title("Human" if y_batch[i] == 1 else "Horse")
    plt.axis('off')
plt.show()

# Create the model using ResNet50 as the base model
base_model = ResNet50(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
base_model.trainable = False

# Add layers on top for binary classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=5)

# Plot accuracy graphs
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
