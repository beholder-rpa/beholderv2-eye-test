import pytesseract
import cv2
import numpy as np
from main import fish_finder

img = cv2.imread('fish_images/Ruby Splashtail_20230604-125050.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh = cv2.threshold(img,170,255,0)

# dialate the image to find clusters of pixels
kernel = np.ones((15,15),np.uint8)
dialate = cv2.dilate(thresh,kernel,iterations = 1)

# find bounding boxes that are at least 25px high
textAreas = []
contours,hierarchy = cv2.findContours(dialate, 1, 2)
for cnt in contours:
  x,y,w,h = cv2.boundingRect(cnt)
  if h > 25 and w > 150:
    textAreas.append((x,y,w,h))

invert = cv2.bitwise_not(thresh)

for (x,y,w,h) in textAreas:
  # get the text from the image
  crop = invert[y:y+h, x:x+w]
  text = pytesseract.image_to_string(crop, config='--psm 6')
  print (text)
  

cv2.imwrite("./test-contour.jpg", invert)
