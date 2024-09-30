Source: https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html

**Good read**
- https://iq.opengenus.org/convolution-filters/
- https://www.kdnuggets.com/2018/04/derivation-convolutional-neural-network-fully-connected-step-by-step.html


# CNN Pseudo Code

## Import Libraries

## Image Preprocessing
1. Load image (chelsea from skimage data)
2. Convert image to grayscale
3. Display image

## Define Filters
1. Create filter bank with 2 filters (3x3 size)
2. Define filters for vertical and horizontal edge detection

## Convolution Layer
Function conv(image, filters):
    For each filter in filters:
        If image or filter has multiple channels:
            Convolve each channel separately
            Sum results of all channels
        Else:
            Perform single channel convolution
    Return feature maps

## ReLU Activation
Function relu(feature_map):
    For each element in feature_map:
        If element < 0:
            Set element to 0
    Return activated feature map

## Max Pooling Layer
Function pooling(feature_map, size, stride):
    For each region in feature_map:
        Take maximum value in the region
    Return downsampled feature map

## CNN Architecture
Create 3 Convolutional Layer, that each contain:
- Create random filters
- Apply convolution
- Apply ReLU activation
- Apply max pooling

## Visualization
- Plot feature maps after each layer