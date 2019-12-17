import tensorflow as tf 
import imgaug.augmenters as iaa
import random
import os 

#image pipeline that will deal with data ingestion and throughput.
class ImagePipeline(object):
    def __init__(self,images,labels,image_size=224):
        self.images = images
        self.labels = labels 
        self.image_size = (image_size,image_size)
        self.ds_size = len(images)

    def parse_data(self,images,labels):
        image_data = tf.io.read_file(images)
        images = tf.image.decode_jpeg(image_data,channels=3) #decode as gray-scale images [W,H,1] 
        images = tf.image.resize(images,self.image_size)
        return images,labels  

    #this works just fine, would just like to keep things 
    #exclusively tf. 
    #def augmentations(self,images,labels):
    #    aug = iaa.OneOf([iaa.PerspectiveTransform(scale=(0.01,0.09)),
    #                     iaa.LogContrast(gain=(0.01,1.0)),
    #                     iaa.Fliplr(0.6)])
    #    return tf.numpy_function(aug.augment_image,[images],images.dtype),labels 
    
    def augmentations(self,images,labels):
        #images = tf.image.random_brightness(images,max_delta=0.3)
        #images = tf.image.random_flip_left_right(images)
        images = tf.image.per_image_standardization(images)
        return images,labels
    
    def validation_generator(self,batch_size,buffer_size):
        data = tf.data.Dataset.from_tensor_slices((self.images,self.labels))
        
        data = data.map(self.parse_data,
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)\
                   .shuffle(buffer_size=buffer_size)\
                   .batch(batch_size)
        return data

    def train_generator(self,batch_size,buffer_size):
        data = tf.data.Dataset.from_tensor_slices((self.images,self.labels)) 
        
        data = data.map(self.parse_data,
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
        data = data.map(self.augmentations,
                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)\
                   .shuffle(buffer_size=buffer_size)\
                   .batch(batch_size)
        return data





