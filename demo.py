from utils import *
from depthnet.model import DepthNet
from pipeline import ImagePipeline
import numpy as np 

#data preparation etc...
train,test,val = image_paths()
normal_train,pneumonia_train = image_classes(train)
normal_test,pneumonia_test = image_classes(test)

normal_train,pneumonia_train = undersample_images(normal_train,pneumonia_train)
normal_test,pneumonia_test = undersample_images(normal_test,pneumonia_test)

train_data = image_label(normal_train,pneumonia_train,shuffle=True)
test_data = image_label(normal_test,pneumonia_test,shuffle=True)

images_train,labels_train = list(zip(*train_data))
images_train = [str(i) for i in images_train]

images_test,labels_test = list(zip(*test_data))
images_test = [str(i) for i in images_test]

images_train = np.array(images_train)
labels_train = np.array(labels_train)

images_test = np.array(images_test)
labels_test = np.array(labels_test)

#generator preparation...

batch_size = 25
buffer_size = 2048

#intialize train generator 
train_pipeline = ImagePipeline(images_train,labels_train)
train_generator = train_pipeline.train_generator(batch_size,buffer_size)

#initialize validation generator
validation_pipeline = ImagePipeline(images_test,labels_test)
validation_generator = validation_pipeline.train_generator(batch_size,buffer_size)

settings = { 'epochs': 10,
             'validation_data' : validation_generator,
             'verbose': 1 }

dn = DepthNet()
dn.initialize()
dn.set_weights() #transfer learning  
dn.fit(generator=train_generator,settings=settings)
dn.save_model('sgd_optimizer')



