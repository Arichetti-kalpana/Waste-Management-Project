import tensorflow as tf
import numpy as np
import os
from sklearn.utils import class_weight 
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# =================================================================
# 1. Configuration
# =================================================================
# Define the number of classes (6 unique folders in your train directory)
NUM_CLASSES = 3

# Define the paths to your dataset
DATASET_PATH = 'C:/Users/arese/Dataset/Dataset/my_dataset/train' 
VALIDATION_PATH = 'C:/Users/arese/Dataset/Dataset/my_dataset/validation'
# NOTE: The paths must be exact and contain the subfolders

# =================================================================
# 2. Data Generators
# =================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# =================================================================
# 3. Class Weight Calculation (FIXES PREDICTION BIAS)
# =================================================================
train_generator.reset() 
classes_array = train_generator.classes 
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes_array),
    y=classes_array
)
class_weights_dict = dict(enumerate(weights))

# CRITICAL: This output MUST be matched in app.py's CLASS_NAMES list.
print("Class indices for app.py (MUST be used in app.py's CLASS_NAMES):", train_generator.class_indices)
print("Calculated Class Weights:", class_weights_dict)

# =================================================================
# 4. Model Definition and Compilation
# =================================================================
base_model = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) 
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Final output is 8 classes

model = Model(inputs=base_model.input, outputs=predictions)
optimizer = Adam(learning_rate=1e-5) 

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =================================================================
# 5. Train and Save the model
# =================================================================
print("Starting training...")
model.fit(
    train_generator,
    epochs=25, # INCREASED EPOCHS for better confidence
    validation_data=validation_generator,
    class_weight=class_weights_dict
)

if not os.path.exists('model'):
    os.makedirs('model')
model.save('model/fine_tuned_vgg16.h5')
print("Model saved as 'model/fine_tuned_vgg16.h5'.")