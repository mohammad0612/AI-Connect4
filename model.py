# Import necessary libraries
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

# Function to load and preprocess the data
def load_data(test_size=0.3):
    # Load dataset
    filename = "data/c4_game_database.csv"
    data = pd.read_csv(filename)

    # Handle NaN values in 'winner' column
    data = data.dropna(subset=['winner'])

    # Separate features and target
    features = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].values

    # Replace target values with 0, 1, 2
    target_mapping = {-1: 0, 0: 1, 1: 2}
    targets = np.vectorize(target_mapping.get)(targets)

    # Reshape features into a 6x7 grid and normalize
    features = features.reshape((-1, 6, 7, 1))
    features = features / 255.0  # Normalization

    # One-hot encode the target variable
    targets = to_categorical(targets, num_classes=3)

    return train_test_split(features, targets, test_size=test_size)

# Function to generate the CNN model
def generate_CNN(lr=0.0001):
    model = Sequential()
    
    # First Convolutional Block
    model.add(Conv2D(128, (4, 4), input_shape=(6, 7, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Block
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Block
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Flattening and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(3, activation='softmax'))  # 3 classes: win, loss, draw

    # Compile the model
    adam_optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_data()

# Create and train the model with the custom architecture
model = generate_CNN(lr=0.0001)  # Adjusted learning rate

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50,  # Increased epochs
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Save the trained model
model.save('connect4_model.h5')
