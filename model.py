# Import necessary libraries
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Function to load and preprocess the data
def load_data():
    # Assuming the last column is the class label
    data = pd.read_csv('data/connect-4.data', header=None)
    
    # Encode the board states
    mapping = {'x': 1, 'o': -1, 'b': 0}
    for col in data.columns[:-1]:  # Exclude the class column
        data[col] = data[col].map(mapping)

    # One-hot encode the target variable
    target_mapping = {'win': 0, 'loss': 1, 'draw': 2}
    data.iloc[:, -1] = data.iloc[:, -1].map(target_mapping)  # Use iloc to reference the last column
    target = to_categorical(data.iloc[:, -1])

    # Reshape the data into a 6x7 grid
    features = data.iloc[:, :-1].values.reshape((-1, 6, 7, 1))  # Reshape data

    # Split the dataset
    return train_test_split(features, target, test_size=0.2, random_state=42)

# Define the CNN model
def create_model():
    model = Sequential()
    
    # First Convolutional Layer
    model.add(Conv2D(64, (4, 4), input_shape=(6, 7, 1), padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Second Convolutional Layer
    model.add(Conv2D(128, (4, 4), padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # Third Convolutional Layer
    model.add(Conv2D(256, (4, 4), padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Flattening and Final Dense Layers
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(3, activation='softmax'))  # 3 for win, loss, draw

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_data()

# Create and train the model
model = create_model()
# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=200,  # Increased epochs
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Save the trained model
model.save('connect4_model.h5')
