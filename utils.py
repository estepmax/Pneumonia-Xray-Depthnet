import matplotlib.pyplot as plt 
import imgaug.augmenters as iaa
from pathlib import Path
import numpy as np 
import random
import glob
import cv2

#initialize base directory as pathlib object and create train,test,val paths 
#to pneumonia images
def image_paths():
    
    base_dir = Path('E:/Downloads/chest-xray-pneumonia/chest_xray')
    val_dir = base_dir / 'val'
    train_dir = base_dir / 'train'
    test_dir = base_dir / 'test'
    
    return train_dir,test_dir,val_dir

#retrieve normal/pneumonia image paths from given directory
def image_classes(dir):
    
    normal = dir / 'NORMAL'
    pneumonia = dir / 'PNEUMONIA'
    
    return normal,pneumonia

#preprocessing steps:
#undersample/oversample -> image_labels -> tf.data
def image_label(class_0,class_1,shuffle=False):
    
    '''
    [args]
        class_0 (Path): list of paths to class_0 images 
        class_1 (Path): list of paths to class_1 images 
    [returns]
        image-label tuple list   
    '''

    class_0_labels = [[1,0]]*len(class_0)
    class_1_labels = [[0,1]]*len(class_1)
    
    labels = class_0_labels + class_1_labels
    image_paths = class_0 + class_1
     
    data = list(zip(image_paths,labels))
    
    if shuffle: #in-place shuffle 
        random.shuffle(data)                
    
    return data

#undersample major class
#note: using random.sample to sample without replacement 
def undersample_images(class_0,class_1):
    '''
    [args]
        class_0 (Path): path to class_0 images 
        class_1 (Path): path to class_1 images 
    [returns]
        list of undersampled image paths 
    '''
    class_0 = list(class_0.glob('*.jpeg'))
    class_1 = list(class_1.glob('*.jpeg'))
    class_0_size = len(class_0)
    class_1_size = len(class_1)
    
    if class_0_size > class_1_size:
        print('class_1 is smaller, undersampling class_0 ({}->{})'
              .format(class_0_size,class_1_size))
        return (random.sample(class_0,k=class_1_size),class_1)
    elif class_1_size > class_0_size:
        print('class_0 is smaller, undersampling class_1 ({}->{})'
              .format(class_1_size,class_0_size))
        return (class_0,random.sample(class_1,k=class_0_size))
    else:
        return (class_0,class_1)

#training_samples = list(normal_train.glob('*.jpeg')) + list(pneumonia_train.glob('.*jpeg'))
#normal_train.glob('*.jpeg') <- how to get images directly

#oversample minor class  
def oversample_images(class_0,class_1):
    '''
    [args]
        class_0 (Path): path to class_0 images 
        class_1 (Path): path to class_1 images 
    [returns]
        list of oversampled image paths 
    '''
    class_0 = list(class_0.glob('*.jpeg'))
    class_1 = list(class_1.glob('*.jpeg'))
    class_0_size = len(class_0)
    class_1_size = len(class_1)

    if class_0_size > class_1_size:
        length = class_0_size - class_1_size
        print('class_0 is larger, generating samples for class_1 ({}->{})'
              .format(class_1_size,class_0_size))
        
        #generate more samples of class_1
        samples = random.choices(class_1,k=length)
        class_1 = class_1 + samples
        
        return (class_0,class_1)
    
    elif class_0_size < class_1_size:
        length = class_1_size - class_0_size
        print('class_1 is larger, generating samples for class_0 ({}->{})'
              .format(class_0_size,class_1_size))

        #generate more samples of class_0
        samples = random.choices(class_0,k=length)
        class_0 = class_0 + samples      
        
        return (class_0,class_1)
    
    else:
        print('class_0 and class_1 are same size')
        return (class_0,class_1)
'''
train,test,val = image_paths()
normal_train,pneumonia_train = image_classes(train)

normal,pneumonia = undersample_images(normal_train,pneumonia_train)

data = image_label(normal,pneumonia,shuffle=True)
print(data)
'''
#image augmentations/transformations 
def image_transformations(path):
    '''
    [args]
        path (Path): direct path to image
    [returns]
        augmented image with given transformations-etc
    '''
    image_aug = iaa.OneOf([iaa.PerspectiveTransform(scale=(0.01,0.09)),
                           iaa.LogContrast(gain=(0.1,1.0)),
                           iaa.Fliplr(0.6)])
    
    img = cv2.imread(str(path),1) #cv2 doesn't like WindowsPath PL object 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(256,256))
    img = image_aug.augment_image(img)
    return img

#sample 15 random images from given directory
def plot_images(dir):
    
    images = list(dir.glob('*.jpeg'))
    print(len(images)) 
    fig = plt.figure(figsize=(10,10))

    for i in range(1,16):
        fig.add_subplot(5,3,i)
        path = random.choice(images)
        img = image_transformations(str(path)) 
        plt.imshow(img,cmap='gray')
    plt.show()

