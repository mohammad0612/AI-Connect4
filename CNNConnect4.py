import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.regularizers import l2
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.layers import Conv2D, Flatten

# Load the dataset
df = pd.read_csv('connect4_data.csv')

# Convert the game state strings to lists of integers
df['Game State'] = df['Game State'].apply(lambda x: list(map(float, x.split(','))))

# Split the dataset into input features (X) and target variable (y)
X = pd.DataFrame(df['Game State'].to_list())
y = df['Best Move']
print(X.head())
print(y.head())

# Encode the target variable
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert the DataFrame to a NumPy array before reshaping
X_train = X_train.to_numpy().reshape(-1, 6, 7, 1)
X_test = X_test.to_numpy().reshape(-1, 6, 7, 1)

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(6, 7, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Save the model
model.save('connect4_cnn_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))