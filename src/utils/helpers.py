#!/usr/bin/env python
# coding: utf-8

# Connecting to google drive for direct access of dataset

# In[ ]:


# mount google drive

from google.colab import drive
drive.mount('/content/drive')


# ## **CONVOLUTIONAL NEURAL NETWORK MODEL**

# Importing necessary modules for performing functions

# In[ ]:


# import modules

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import plot_model


# Loading the dataset along with preprocessing images for a better clear image and splitting dataset into train and validation part

# In[ ]:


# load dataset and set class labels

astronomy_data = '/content/drive/MyDrive/FinalProjectWork/TRAIN'
astronomy_class_labels = ['star', 'galaxy']



# preprocess images

star_galaxy_image_size = (64, 64)
astronomy_batch_size = 32

    rescale=1./255,
    rotation_range = 45,  # range of rotation angles (in degrees)
    zoom_range=0.2, # zooming image by 20%
    horizontal_flip=True, # flipping image horizontally
    validation_split=0.2 # split 20% of dataset into for validation
)



# train CNN model from dataset features

    astronomy_data,
    target_size=star_galaxy_image_size,
    batch_size=astronomy_batch_size,
    class_mode='binary',
    subset='training'
)

    astronomy_data,
    target_size=star_galaxy_image_size,
    batch_size=astronomy_batch_size,
    class_mode='binary',
    subset='validation'
)


# In[ ]:


augmented_img = astronomy_datagen.random_transform(astronomy_train_generator[0][0][0])
plt.title('Final Augmented Image')
plt.axis('off')
plt.show()


# In[ ]:


# Assuming train_generator contains your normal star-galaxy images
normal_img = astronomy_train_generator[0][0][0]
plt.title('Normal Image Before Rotation')
plt.axis('off')
plt.show()


# In next step, CNN model is build by use of Dropout regulariser and displayed

# In[ ]:


# CNN model build



# define input shape

input_shape = (star_galaxy_image_size[0], star_galaxy_image_size[1], 3)



# define input layer

inputs = layers.Input(shape=input_shape, name='Input')
pool_1 = layers.MaxPool2D(pool_size=(2, 2), name='MaxPool2D_1')(conv_1)
pool_2 = layers.MaxPool2D(pool_size=(2, 2), name='MaxPool2D_2')(conv_2)
pool_3 = layers.MaxPool2D(pool_size=(2, 2), name='MaxPool2D_3')(conv_3)
flatten = layers.Flatten(name='flatten')(pool_3)
dense_1 = layers.Dense(128, activation='relu', name='dense_1')(flatten)
dropout = layers.Dropout(0.5, name='Dropout')(dense_1)
output = layers.Dense(1, activation='sigmoid', name='output')(dropout)



# define model

astronomy_model = models.Model(inputs=inputs, outputs=output, name='astronomy_model')



# display model

astronomy_model.summary()


# In[ ]:


# display CNN model

plot_model(astronomy_model, show_shapes=True, show_layer_names=True)


# Now, the CNN model that was build above will be compiled and trained for 10 epochs

# In[ ]:


# compile model

astronomy_model.compile(optimizer='Adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])



# run model with 10 epochs

history = astronomy_model.fit(
    astronomy_train_generator,
    steps_per_epoch = astronomy_train_generator.samples // astronomy_batch_size,
    epochs = 10,
    validation_data = astronomy_validation_generator,
    validation_steps = astronomy_validation_generator.samples // astronomy_batch_size
)


# Evaluating the model along with plotting accuracy and loss curves and values as well

# In[ ]:


# model evaluation

    astronomy_data,
    target_size=star_galaxy_image_size,
    batch_size=32,
    class_mode='binary',
    shuffle=False
)



# plot accuracy and loss curves

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('Model Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.title('Model Accuracy')
plt.legend(loc='upper right')
plt.show()


# In[ ]:


# display loss and accuracy of cnn model

astronomy_loss, astronomy_accuracy = astronomy_model.evaluate(astronomy_test_generator)
print(f"CNN Model Loss: {astronomy_loss:.2f}")
print(f"CNN Model Accuracy: {astronomy_accuracy:.2f}")


# Here, the prediction of classified images is performed and correctly classified images are displayed along with their labels

# In[ ]:


# display correctly classified images

x_test, y_test = astronomy_validation_generator.next()
y_pred = astronomy_model.predict(x_test)

fig, axes = plt.subplots(5, 5, figsize=(12, 12))
fig.subplots_adjust(hspace=0.8, wspace=0.8)
for i, ax in enumerate(axes.flat):
  ax.imshow(x_test[i])
  if y_pred[i][0] < 0.5: # set conditions for classification of stars and galaxies
    predict_label = 'Star'
  else:
    predict_label = 'Galaxy'

  if y_test[i] == 0:
    true_label = 'Star'
  else:
    true_label = 'Galaxy'

  title = f'True: {true_label}\nPrediction: {predict_label}'
  ax.set_title(title, fontsize=10)
  ax.axis('off')
plt.show()


# Performing metrics insights like Precision, Recall, F1-score, Confusion Matrix and AUC ROC Curve on CNN model

# In[ ]:


# evaluate model

y_true = astronomy_test_generator.classes
y_pred = astronomy_model.predict(astronomy_test_generator)
y_pred_classes = np.round(y_pred).flatten().astype(int)



# computing confusion matrix




# calculating other metrics

astronomy_precision = precision_score(y_true, y_pred_classes)
astronomy_recall = recall_score(y_true, y_pred_classes)
astronomy_f1 = f1_score(y_true, y_pred_classes)



# display values of metrics

print("Confusion Matrix of CNN is: ", astronomy_cm)
print("Precision of CNN is: ", astronomy_precision)
print("Recall of CNN is: ", astronomy_recall)
print("F1 score of CNN is: ", astronomy_f1)



# plot confusion matrix

plt.figure(figsize=(6, 6))
sns.heatmap(astronomy_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=astronomy_class_labels, yticklabels=astronomy_class_labels)
plt.title("Confusion Matrix of CNN Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()



# plt ROC curve

astronomy_roc_auc = roc_auc_score(y_true, y_pred)

plt.figure(figsize=(6, 6))
         label="ROC Curve (AUC = %0.2f)" % astronomy_roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic(ROC)")
plt.legend(loc="lower right")
plt.show()


# 
# ## **RANDOM FOREST CLASSIFIER MODEL**

# Importing necessary modules

# In[ ]:


# import modules

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import random


# Dataset is loaded by use of function along with changing images to grayscale and extracting features from them

# In[ ]:


# load dataset

def astronomy_load_data(image_dir):
  X_astronomy_data = []
  y_astronomy_label = []
  astronomy_classes = os.listdir(image_dir)
  for astronomy_class_name in astronomy_classes:
    astronomy_class_path = os.path.join(image_dir, astronomy_class_name)
    if os.path.isdir(astronomy_class_path):
      astronomy_class_label = 0 if astronomy_class_name == "star" else 1
      for image_file in os.listdir(astronomy_class_path):
        astronomy_image_path = os.path.join(astronomy_class_path, image_file)
        astronomy_image = Image.open(astronomy_image_path).convert('L') # convert image to grayscale
        astronomy_image_array = np.array(astronomy_image)
        X_astronomy_data.append(astronomy_image_array.flatten()) # flattening the image for creating feature vectors
        y_astronomy_label.append(astronomy_class_label)
  return np.array(X_astronomy_data), np.array(y_astronomy_label)

astronomy_image_dir = '/content/drive/MyDrive/FinalProjectWork/TRAIN'
X_astronomy_data, y_astronomy_label = astronomy_load_data(astronomy_image_dir)


# RF classifier model is trained with test size of 20% and remaining 80% is used for training purposes

# In[ ]:


# train the RF model

X_train, X_test, y_train, y_test = train_test_split(X_astronomy_data,
                                                    y_astronomy_label,
                                                    test_size=0.2,
                                                    random_state=42)


# The model is built with 100 number of trees so that proper feature extraction is performed

# In[ ]:


# build RF model

astronomy_rf = RandomForestClassifier(n_estimators=100, random_state=42)


# RF classifier model evaluation is done in following part of code and classification report containing Precision, Recall, F1-score and Support has been generated

# In[ ]:


# evaluate RF model



# predict the outcome
astronomy_y_pred = astronomy_rf.predict(X_test)



# calculate RF model accuracy

print("RF Model Accuracy is: ", astronomy_rf_accuracy)



# calculate RF model classification report

print("RF Model Classification Report: \n", astronomy_class_report)


# Along with this, Confusion Matrix has been plotted

# In[ ]:


# calculate and display confusion matrix

print("\n") # extra space below after confusion matrix



# plot confusion matrix

plt.figure(figsize=(8, 6))
plt.title("RF Model Confusion Matrix")
plt.xticks([0.5, 1.5], ["Predicted Star", "Predicted Galaxy"]) # set position for "x" labels
plt.yticks([0.5, 1.5], ["True Star", "True Galaxy"]) # set position for "y" labels
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# Randomly classified images has been displayed along with labels on top of them

# In[ ]:


# display correctly classified images



# randomly select subest for display of images
astronomy_samples = 25
random_indices = random.sample(range(len(y_test)), astronomy_samples)
astronomy_X = X_test[random_indices]
astronomy_y_true = y_test[random_indices]
astronomy_y_visualize_pred = astronomy_y_pred[random_indices]



# list for storing correctly classfied images

rf_correctly_classified = []



# interpret image shape from dataset

rf_image_shape = (int(np.sqrt(X_astronomy_data.shape[1])),
                  int(np.sqrt(X_astronomy_data.shape[1])))



# display samples with labels

plt.figure(figsize=(12, 12))
for i in range(astronomy_samples):
  plt.subplot(5, 5, i + 1)
  plt.title(f"True: {'Star' if astronomy_y_true[i] == 0 else 'Galaxy'}\nPred: {'Star' if astronomy_y_pred[i] == 0 else 'Galaxy'}")
  plt.axis('off')
  if astronomy_y_true[i] == astronomy_y_pred[i]:
    rf_correctly_classified.append(i)

plt.tight_layout()
plt.show()


# AUC ROC curve is displayed which depicts the behaviour and accuracy of the model overall

# In[ ]:


# display AUC ROC Curve for RF Model



# calculate predicted probabilties for positive class

rf_y_pred_prob = astronomy_rf.predict_proba(X_test)[:, 1]



# calculate ROC Curve

                                                                rf_y_pred_prob)



# Calculate AUC for ROC Curve

rf_auc = roc_auc_score(y_test, rf_y_pred_prob)



# plot the results

plt.figure(figsize=(8, 8))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show

