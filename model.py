import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths to your image directory and CSV files
image_directory = "C:\\INTELLIGENT\\dataset\\data\\data" 
train_csv_path = "C:\\INTELLIGENT\\dataset\\train.csv"  
test_csv_path = "C:\\INTELLIGENT\\dataset\\train.csv"    

# Load training data
train_data = pd.read_csv(train_csv_path)
train_images = []
train_labels = []

# Create a mapping of class names to integers
label_dict = {name: index for index, name in enumerate(train_data['class'].unique())}

# Load training images and labels
for index, row in train_data.iterrows():
    img_path = os.path.join(image_directory, row['image_name'] + '.png')  # Add .png extension
    img = load_img(img_path, target_size=(224, 224))  # Resize images as needed
    img_array = img_to_array(img) / 255.0  # Normalize images
    train_images.append(img_array)
    train_labels.append(label_dict[row['class']])

# Convert lists to NumPy arrays for training data
X_train = np.array(train_images)
y_train = to_categorical(np.array(train_labels))

# Data augmentation for training set
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of training data for validation
)

# Flow training images in batches
train_generator = data_gen.flow(X_train, y_train, subset='training')
validation_generator = data_gen.flow(X_train, y_train, subset='validation')

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(label_dict), activation='softmax'))  # Output layer for the number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
model.fit(train_generator, 
          validation_data=validation_generator, 
          epochs=30,  # Increase epochs if needed
          callbacks=[checkpoint, early_stopping])

# Load testing data
test_data = pd.read_csv(test_csv_path)
test_images = []

# Load testing images
for index, row in test_data.iterrows():
    img_path = os.path.join(image_directory, row['image_name'] + '.png')  # Add .png extension
    img = load_img(img_path, target_size=(224, 224))  # Resize images as needed
    img_array = img_to_array(img) / 255.0  # Normalize images
    test_images.append(img_array)

# Convert lists to NumPy arrays for testing data
X_test = np.array(test_images)

# Predict classes for the test set
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted classes back to original labels
predicted_labels = [list(label_dict.keys())[list(label_dict.values()).index(predicted_class)] for predicted_class in predicted_classes]

# Save predictions to a CSV file
output_df = pd.DataFrame({'image_name': test_data['image_name'], 'predicted_class': predicted_labels})
output_df.to_csv('predictions.csv', index=False)

# Evaluate the model on the test set (if you have true labels for evaluation)
#test_loss, test_accuracy = model.evaluate(X_test, y_test)
#print(f'Test accuracy: {test_accuracy:.2f}')
# Save the model
model.save('your_model.h5')  # Save the final model
