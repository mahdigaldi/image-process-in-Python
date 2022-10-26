# import cv2
# import numpy as np
# img = cv2.imread('pic1.jpg')
# def increase_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     lim = 255 - value
#     v[v > lim] = 255
#     v[v <= lim] += value

#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     cv2.imwrite("change.jpg",img)

# frame = increase_brightness(img, value=20)

# new way


# import cv2
# import matplotlib.pyplot as plt
# img=cv2.imread('pic2.png')
# gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage import img_as_float
img=cv2.imread('fig1.jpg')
img=img_as_float(img)
# plt.imshow(img,cmap='gray')
# plt.show()
# c=2
# im_tr=c*np.log(1+img)
# fig,ax=plt.subplots(2,2,figsize=(7,7))
# ax[0,0].imshow(img,cmap='gray')
# ax[0,1].hist(img.ravel(),bins=256)
# ax[1,0].imshow(im_tr,cmap='gray')
# ax[1,1].hist(im_tr.ravel(),bins=256)
# plt.show()

#newway 

gama=3
im_tr=img**gama
fig,ax=plt.subplots(2,2,figsize=(7,7))
ax[0,0].imshow(img,cmap='gray')
ax[0,1].hist(img.ravel(),bins=256)
ax[1,0].imshow(im_tr,cmap='gray')
ax[1,1].hist(im_tr.ravel(),bins=256)
plt.show()