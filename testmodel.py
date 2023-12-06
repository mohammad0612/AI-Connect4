import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pandas as pd

def generate_CNN():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(6, 7, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 classes: win, loss, draw

    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

def load_shape_ttsplit(filename, test_size=0.3):
    data = pd.read_csv('data/c4_game_database.csv')
    data = data.dropna(subset=['winner'])
    data = data.to_numpy()
    X0 = data[:, :-1].copy()
    y0 = data[:, -1].copy()
    print("Before conversion:", np.unique(y0))
    y0 = (y0 + 1) // 2
    print("After conversion:", np.unique(y0))   
    X0 = X0.reshape(X0.shape[0], 6, 7, 1)
    assert not np.any(np.isnan(y0)), "NaN values found in labels"
    assert np.all((y0 >= 0) & (y0 < 3)), "Invalid label values found"       
    return train_test_split(X0, y0, test_size=test_size)

X_train, X_test, y_train, y_test = load_shape_ttsplit('data/c4_game_database.csv')

model = generate_CNN()

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save the model
model.save('connect4_testmodel.h5')

