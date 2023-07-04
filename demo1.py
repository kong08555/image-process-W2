import numpy as np
import cv2 as cv
#read image
imgA = cv.imread("pic-work/work2.jpg",cv.IMREAD_GRAYSCALE)
imgB = cv.imread("pic-work/work3.jpg",cv.IMREAD_GRAYSCALE)

histB = cv.calcHist([imgB],[0],None,[256],[0, 256])

imgAtoB = cv.equalizeHist(imgA,histB)


cv.imwrite('imgA.png', imgA)
cv.imwrite('imgB.png', imgB)
cv.imwrite('imgAtoB.png', imgAtoB)