import math
import os.path
import json
import skimage
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.

class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):

        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        with open(label_path) as json_file:
            self.labels = json.load(json_file)
        
        self.images = [file.split(".")[0] for file in os.listdir(file_path)]

        if self.shuffle:
            np.random.shuffle(self.images)

        self.num_batches = math.ceil(len(self.images)/self.batch_size)
        self.current_batch = 0
        self.curr_epoch = 0

        self.batched_images = self.split_array_into_batches(self.images, self.batch_size)

    @staticmethod
    def split_array_into_batches(array, batch_size):
        num_batches = len(array) // batch_size
        remainder = len(array) % batch_size

        full_batches = np.array_split(array[:num_batches * batch_size], num_batches)

        if remainder > 0:
            last_batch = np.concatenate((array[-remainder:], array[:batch_size - remainder]))
            full_batches.append(last_batch)

        return full_batches
    
    def to_resized_und_augmented_np_image_array(self, image_files):
        return np.array([self.augment(skimage.transform.resize(np.load(f"{self.file_path}/{file}.npy"), self.image_size)) for file in image_files])

    def next(self):

        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        if self.current_batch == self.num_batches:
            self.current_batch = 0
            self.curr_epoch += 1

            if self.shuffle:
                np.random.shuffle(self.images)
                self.batched_images = self.split_array_into_batches(self.images, self.batch_size)

        images, labels = self.batched_images[self.current_batch], [self.labels[l] for l in self.batched_images[self.current_batch]]

        self.current_batch += 1

        return self.to_resized_und_augmented_np_image_array(images), labels

    def augment(self,img):
        
        if self.mirroring:
            img = np.flip(img)
        
        if self.rotation:
            angle = np.random.choice((90,180,270))
            img = rotate(img, angle=angle)

        return img

    def current_epoch(self):
        return self.curr_epoch

    def class_name(self, x):
        return self.class_dict[self.labels[x]]
    
    def draw(self):
        pass
    
    def show(self):
    
        images_to_plot = self.to_resized_und_augmented_np_image_array(self.batched_images[self.current_batch])

        for i in range(self.batch_size):
            plt.subplot(1, self.batch_size+1, i+1)
            plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
            plt.imshow(images_to_plot[i])
