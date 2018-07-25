import numpy as np
import cv2

img = cv2.imread("data/credits/HW2pRgqHR6GnYhmmFIGTgXrnaQll5Jcwv0hnzfyT1sfRLSTGt74t3l8clFlAWUPp0brDHz9EZ84X4F9cDLrBxS014AAAAAElFTkSuQmCC.jpg")

image_array = np.asarray(img)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

small = cv2.resize(img, (200, 200))

cv2.imshow('image', small)
cv2.waitKey(0)
cv2.destroyAllWindows()

