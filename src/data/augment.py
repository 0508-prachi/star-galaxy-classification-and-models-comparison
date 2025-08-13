from tensorflow.keras.preprocessing.image import ImageDataGenerator
astronomy_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
astronomy_train_datagen = ImageDataGenerator(