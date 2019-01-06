# Day or Night Image Classification
**simple image classification in python**

# Simple Image Classifier
### Steps:

1. Load the data set
2. Visualize the data inputs
3. Pre-process the data
    - standardize inputs and outputs
    
4. Visualise standardized data
5. Convert image to HSV
6. Calculate image intensity average
7. Calculate average upper and lower bounds for each catagory
8. TODO: finish the classifier code



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
- Shape: (600, 1100, 3)
- Label [1 = day, 0 = night]: 1

![Example Image](https://github.com/CodeSenpii/ImageClassification/blob/master/day_image1.png)

# Feature extraction process

** In this example we will be using the brightness feature of the image to do the classification. **

** Its really simple! All we do is calulate the average brighness of the photo to determing whether it is a night time image or a daytime image **

## We wll be using the "HSV color space" technique as part of the process


### Step 1:  convert the RGB image to HSV


### Let us see what happends when an image in conveted to HSV 

 - ** get the image **
 
 ```python
 # EXAMPLE 1

# first get the image

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]
 ```
 
 # Convert the image from RGB to HSV
 
 ```python
 # Example 1 cont'

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label: ' + str(test_label))
 ```
 
 Label: 1
 
 ## Seperate the HSV Channels
 
 ```python
 # Example 1 cont'
# Seperating the HSV channels

# HSV channels
h = hsv[:,:,0]  # channel 0
s = hsv[:,:,1]  # channel 1
v = hsv[:,:,2]  # channel 2
 ```
 
 ## Display the image in a Row
 
 ```python
 # Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')
```
![Example Image](https://github.com/CodeSenpii/ImageClassification/blob/master/imag1.png)

## Finding the average brightness of an image
```python
# Find the average Value or brightness of an image
def avg_brightness(rgb_image):
    
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(hsv[:,:,2])
    
    area = 600 * 1100.0
    #Calculate the average brightness using the area of the image
    # and the sum calculated above
    avg = sum_brightness/area
    
    return avg
```

# Get avg values - Testing 

```python
# Testing average brightness levels
# Look at a number of different day and night images and think about 
# what average brightness value separates the two types of images

# As an example, a "night" image is loaded in and its avg brightness is displayed
image_num = 54

test_im = STANDARDIZED_LIST[image_num][0]

avg = avg_brightness(test_im)
print('Avg brightness: ' + str(avg))
plt.imshow(test_im)
```
Avg brightness: 87.22738787878788
![Example Image](https://github.com/CodeSenpii/ImageClassification/blob/master/night_image1.png)

```python
def image_ranges():
''' This function finds the upper and lower range of the avg brightness for each catagory '''
''' These values gives a good indication of how to set the classifier '''
''' @Author Kieyn Parks
    
    # Ranges
    lower_day = 0
    upper_day = 0
    counter1 = 0
    counter2 = 0
        
    lower_night = 0
    upper_night = 0
        
    for image in STANDARDIZED_LIST:
        
        img = image[0]
        lab = image[1]    
        
        if lab == 1:
            # Convert image to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # Add up all the pixel values in the V channel
            avg = np.sum(hsv[:,:,2]) / (600*1100.0)
            
            if lower_day > avg or lower_day == 0:
                lower_day = avg
            
            if upper_day < avg or upper_day == 0:
                upper_day = avg
                
            counter1 += 1
            
        elif lab == 0:
             # Convert image to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # Add up all the pixel values in the V channel
            avg = np.sum(hsv[:,:,2]) / (600*1100.0)
            
            if lower_night > avg or lower_night == 0:
                lower_night = avg
            
            if upper_night < avg or upper_night == 0:
                upper_night = avg
            
            counter2 += 1
    
            
    print("Day Range: ", lower_day,"-",upper_day, ":", counter1)
    print("Night Range: ", lower_night,"-",upper_night, ":", counter2) 
image_ranges()         
```
- **Day Range:  98.89392727272727 - 201.6465924242424 : 45**
- **Night Range:  8.24230909090909 - 98.99117727272727 : 46**

## Classifier
```python
# Not implimented

if avg > 120:
    print("day")
else:
    print("night")
```
