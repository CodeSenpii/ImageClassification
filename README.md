# Day or Night Image Classification
**simple image classification in python**

# Simple Image Classifier
### Steps:

1. Load the data set
2. Visualize the data inputs
3. Pre-process the data
    - standardize inputs and outputs
    
4. Visualise standardized data



## The helper function below is responsible for getting the images from the directory
## and properly labeling each image. 
```python
# Helper functions  - Udacity.com- Intro To Self Driving Cars

import os
import glob # library for loading images from a directory
import matplotlib.image as mpimg



# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["day", "night"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list

```
## Get resources

```python
import cv2 # computer vision library


import numpy as np # operating on image
import matplotlib.pyplot as plt # drawing image
#import matplotlib.image as mpimg # getting image

# display image inline notebook
%matplotlib inline  
```
### Get the directory containing images

```python
# Image data directories
image_dir_training = "C:\\Users\super\\Desktop\\Self Driving Cars\\Day and Night Images"
```

## Load up the images using the function "load_dataset"

```python
# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = load_dataset(image_dir_training)
```

## Standardizing the images
1. standardize the input (image)
2. standardize the output/label from catagorical data to numeric data.

### ** To standardize the images we will make them all the same size/dimemsions **

- function standardize_input()

```python
# This function should take in an RGB image and return a new, standardized version
# 600 height x 1100 width image size (px x px)
def standardize_input(image):
    
    # Resize image and pre-process so that all "standard" images are the same size  
    standard_im = cv2.resize(image, (1100, 600))
    
    return standard_im
```
## We will standardize the output/labels

- convert the catagorical data "day" "nigth" to a more usable numerical data 0/1

```python
# Examples: 
# encode("day") should return: 1
# encode("night") should return: 0
def encode(label):
        
    numerical_val = 0
    if(label == 'day'):
        numerical_val = 1
    # else it is night and can stay 0
    
    return numerical_val
```

## This function will take the Raw data list and coordinate the standardization process it will return the final list

```python
# using both functions above, standardize the input images and output labels
def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # Create a numerical label
        binary_label = encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, binary_label))
        
    return standard_list
```
## Lets Construct to standardized list

```python
# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)
```

# Visually verify the images have been converted successfully
```python
# Display a standardized image and its label

# Select an image by index
image_num = 0
selected_image = STANDARDIZED_LIST[image_num][0]
selected_label = STANDARDIZED_LIST[image_num][1]

# Display image and data about it
plt.imshow(selected_image)
print("Shape: "+str(selected_image.shape))
print("Label [1 = day, 0 = night]: " + str(selected_label))
```
**Correct dimensions**
Shape: (600, 1100, 3)
Label [1 = day, 0 = night]: 1

![Example Image](https://github.com/CodeSenpii/ImageClassification/blob/master/day_image1.png)

