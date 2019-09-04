
# Fruits-360 - Transfer Learning using Keras and ResNet-50

* This notebook is a brief application of transfer learning on the Fruits-360 [Dataset](https://www.kaggle.com/moltean/fruits). 
* This data set consists of 42345 images of 64 fruits.
* We compare the Transfer learning approach to the regular approach.
* Accuracy of 98.44% is achieved within 2 epochs.


### **Contents:**

*  **1. Brief Explanation of Transfer Learning**
*  **2. Transfer Learning using Kaggle Kernels**
*  **3. Reading and Visualizing the Data**   
*  **4. Building and Compiling the Models**    
*  **5. Training and Validating the Pretrained Model** 
*  **6. Training and Validating Vanilla Model**
*  **7. Comparison between Pretrained Models and Vanilla Model**


```python
########################################### Library Import ###########################################

# 합성곱 연산 : https://excelsior-cjh.tistory.com/180
# 기본적인 CNN 코드에 대해서는 : https://subinium.github.io/Keras-5-1/

import os
from os import listdir, makedirs
from os.path import join, exists, expanduser

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
# Any results you write to the current directory are saved as output.

########################################### Create a folder for Pretrained Model ###########################################

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
    
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/

print("Available Pretrained Models:\n")
!ls ~/.keras/models


########################################### Train & Valid Set ###########################################

# dimensions of our images.
img_width, img_height = 224, 224 
# we set the img_width and img_height according to the pretrained models we are
# going to use. The input size for ResNet-50 is 224 by 224 by 3.

train_data_dir = '../input/fruits/fruits-360_dataset/fruits-360/Training/'
validation_data_dir = '../input/fruits/fruits-360_dataset/fruits-360/Test/'
nb_train_samples = 31688
nb_validation_samples = 10657
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255, # Scaling 작업
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Imagegenerator 자세한 설명은 이 주소 참고 https://tykimos.github.io/2017/03/08/CNN_Getting_Started/

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# flow_from_directory는 폴더 형식의 데이터를 불러올 때 사용

# 데이터 확인
from keras import layers, models
from keras.preprocessing import image
from glob import glob
import numpy as np

print('number of image files', len(image_files))
image_files = glob(train_data_dir + '/*/*.jp*g') # 해당 경로에 확장자를 jpg로 갖고 있는 파일 이름 모두 갖고오기
plt.imshow(image.load_img(np.random.choice(image_files)))


########################################### Import Pretrained Model ###########################################

#import inception with pre-trained weights. do not include fully #connected layers
inception_base = applications.ResNet50(weights='imagenet', include_top=False)
# application은 이미 train된 것에서 top을 제외하고 network 갖고 올 수 있게 하는 함수

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)
# global average pooling에 대해서는 여기 참고 https://poddeeplearning.readthedocs.io/ko/latest/CNN/CAM%20-%20Class%20Activation%20Map/

# add a fully-connected layer
x = Dense(512, activation='relu')(x)

# and a fully connected output/classification layer
predictions = Dense(118, activation='softmax')(x)

# create the full network so we can train on it
inception_transfer = Model(inputs=inception_base.input, outputs=predictions)

inception_transfer.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


```


```python
######### Keras 식으로 간결하게 표현해보기 

inception_base = applications.ResNet50(weights='imagenet', include_top=False)

model = models.Sequential()
model.add(inception_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(118, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
batch_size = 64
with tf.device("/device:GPU:0"):
    history_pretrained = model.fit_generator(
    train_generator,steps_per_epoch = int(nb_train_samples / batch_size),
    epochs=5, shuffle = True, verbose = 1,
    validation_data = validation_generator, validation_steps = int(nb_validation_samples / batch_size))
    
    
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_pretrained.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history_pretrained.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()
```


```python
#import inception with pre-trained weights. do not include fully #connected layers
inception_base = applications.ResNet50(weights='imagenet', include_top=False)
# application은 이미 train된 것에서 top을 제외하고 network 갖고 올 수 있게 하는 함수

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)
# global average pooling에 대해서는 여기 참고 https://poddeeplearning.readthedocs.io/ko/latest/CNN/CAM%20-%20Class%20Activation%20Map/

# add a fully-connected layer
x = Dense(512, activation='relu')(x)

# and a fully connected output/classification layer
predictions = Dense(118, activation='softmax')(x)

# create the full network so we can train on it
inception_transfer = Model(inputs=inception_base.input, outputs=predictions)

inception_transfer.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


```


```python

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
batch_size = 64
with tf.device("/device:GPU:0"):
    history_pretrained = inception_transfer.fit_generator(
    train_generator,steps_per_epoch = int(nb_train_samples / batch_size),
    epochs=5, shuffle = True, verbose = 1,
    validation_data = validation_generator, validation_steps = int(nb_validation_samples / batch_size))
    
    
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_pretrained.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history_pretrained.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()
```

## 1. Transfer Learning

In transfer learning, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, meaning suitable to both base and target tasks, instead of specific to the base task.

Lisa Torrey and Jude Shavlik in their chapter on transfer learning describe three possible benefits to look for when using transfer learning:

* Higher start. The initial skill (before refining the model) on the source model is higher than it otherwise would be.
* Higher slope. The rate of improvement of skill during training of the source model is steeper than it otherwise would be.
* Higher asymptote. The converged skill of the trained model is better than it otherwise would be.

<center><img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Three-ways-in-which-transfer-might-improve-learning.png"></center>


Basically, we take a pre-trained model (the weights and parameters of a network that has been trained on a large dataset by somebody else) and “fine-tune” the model with our own dataset. The idea is that this pre-trained model will either provide the initialized weights leading to a faster convergence or it will act as a fixed feature extractor for the task of interest.



These two major transfer learning scenarios look as follows:

* Finetuning the convnet: Instead of random initializaion, we initialize the network with a pretrained network, like the one that has been trained on a large dataset like imagenet 1000. Rest of the training looks as usual. In this scenario the entire network needs to be retrained on the dataset of our interest

* ConvNet as fixed feature extractor: Here, we will freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained.

In this notebook we will demonstrate the first scenario.


## 2. Transfer Learning using Kaggle Kernels

### Using the Keras Pretrained Models dataset
Kaggle Kernels cannot use a network connection to download pretrained keras models. This [Dataset](https://www.kaggle.com/moltean/fruits) helps us to use our favorite pretrained models in the Kaggle Kernel environment.

All we have to do is to copy the pretrained models to the cache directory (~/.keras/models) where keras is looking for them.

## 3. Reading and Visualizing the Data
### Reading the Data

Like the rest of Keras, the image augmentation API is simple and powerful. We will use the **ImageDataGenerator** to fetch data and feed it to our network

Keras provides the **ImageDataGenerator** class that defines the configuration for image data preparation and augmentation. Rather than performing the operations on your entire image dataset in memory, the API is designed to be iterated by the deep learning model fitting process, creating augmented image data for you just-in-time. This reduces your memory overhead, but adds some additional time cost during model training.

The data generator itself is in fact an iterator, returning batches of image samples from the directory when requested. We can configure the batch size and prepare the data generator and get batches of images by calling the **flow_from_directory()** function.
