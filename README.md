# PlaceDeep365
Detecting Places by Deep Learning. 
In this Deep Learning project I decided to detect three important places in Iran(Azadi Square, The shrine of Imam Reza, Taq-e Bostan)
for this detection, I have to use fine tuning and a model named VGG16PLACE with places weight. 
# Contents:
This repository contains code for the following [Keras](https://keras.io/) models:
  VGG16-places365
## Code Description
Using VGG16PLACE with places weight and making 5 last layers *Unfreeze*
```python
from vgg16_places_365 import VGG16_Places365
base_model = VGG16_Places365(include_top=False, weights='places', input_shape=(150, 150, 3))
for layer in base_model.layers[:12]:
    layer.trainable = True
for layer in base_model.layers[12:]:
    layer.trainable = False
```
Connecting base_model to the 256 node (Dense Layer) 
```python
from keras import models
from keras import layers
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
```
Using imagedatagenerator to avoid Overfit. 
what is [overfiting?](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)

```python
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)
 
test_datagen = ImageDataGenerator(rescale=1./255)
 
train_dir = './Places/train'
validation_dir = './Places/validation'
 
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 108x192
        target_size=(150, 150),
        batch_size=20,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
 
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')
```
Compiling the network and Training process. Learning Rate is very little. In fine Tuning you have to use a little LR, because we almost get to the local min.
```python
model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=2e-5),
            metrics=['acc'])
num_of_sample_train = 600
num_of_sample_validation = 150

batch_size = 20
history = model.fit_generator(
    train_generator,
    steps_per_epoch=num_of_sample_train/batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=num_of_sample_validation/batch_size)
```
Because of some errors I had to write the preproccessing stuffs from scrath(just copy the code, don't think about it)
```python
from keras import backend as K
def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.common.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    
    if dim_ordering == 'th':
        x[:, 0, :, :] -= 104.006
        x[:, 1, :, :] -= 116.669
        x[:, 2, :, :] -= 122.679
        x = x[:, ::-1, :, :]
    
    else:
        x[:, :, :, 0] -= 104.006
        x[:, :, :, 1] -= 116.669
        x[:, :, :, 2] -= 122.679
        x = x[:, :, :, ::-1]
    return x
```
Downloading a Pic of Taq-e Bostan and test our CNN_PLACE365. rezise the image to (150 * 150) and normalization with preprocess_input 
```python
from keras.preprocessing import image
from keras.models import load_model
import keras
import numpy as np
#Tagh_Bostan picture
ImageAddress = 'E:/2.jpg'
img = image.load_img(ImageAddress, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
```
Predictions variable gives the predictions of all 3. we need the max of the predictions.
```python
prediction = model.predict(x)
y_classes = prediction.argmax()
labels = ['Azadi', 'Emam_Reza', 'Tagh_Bostan']
for i, p in enumerate(prediction[0]):
    print('%s Probability: \t %f' % (labels[i], p))

```
