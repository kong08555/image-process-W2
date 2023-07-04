import numpy as np
import cv2 as cv

imgA = cv.imread("image/work2.jpg", cv.IMREAD_GRAYSCALE)

user_specific_value = int(input("Enter mean value: "))
window_size = int(input("Enter window size: "))
gamma_more = float(input("Enter the gamma value for γ < 1: "))
gammma_less = float(input("Enter the gamma value for γ > 1: "))

block_size = (window_size, window_size)  
stride = (block_size[0] // 2, block_size[1] // 2)  

blocks_y = (imgA.shape[0] - block_size[0]) // stride[0] + 1
blocks_x = (imgA.shape[1] - block_size[1]) // stride[1] + 1

blocks = []
block_histograms = []
block_means = []

for y in range(blocks_y):
    for x in range(blocks_x):
        block = imgA[y * stride[0]: y * stride[0] + block_size[0], x * stride[1]: x * stride[1] + block_size[1]]
        blocks.append(block)
        
        hist = cv.calcHist([block], [0], None, [256], [0, 256])
        block_histograms.append(hist)
        
        value_mean = np.mean(block)
        block_means.append(value_mean)
        
if value_mean < user_specific_value:
    block_gamma = np.power(block / 255.0 , gammma_less) * 255.0
else:
    block_gamma = np.power(block / 255.0, gamma_more) * 255.0
    

block_gamma = block_gamma.astype(np.uint8)

cv.imwrite(f"divide_img/block_{y}_{x}.png",block_gamma)
    
for i, hist in enumerate(block_histograms):
    value_mean = block_means[i]
    print(f"Block {i}: Histogram: {hist}, Mean Value: {value_mean}")
