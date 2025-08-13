from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
conv_1 = layers.Conv2D(32, (3, 3), activation='relu', name='conv_1')(inputs)
conv_2 = layers.Conv2D(64, (3, 3), activation='relu', name='conv_2')(pool_1)
conv_3 = layers.Conv2D(128,(3, 3), activation='relu', name='conv_3')(pool_2)