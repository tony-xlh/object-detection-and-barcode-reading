import cv2
import numpy as np
image = cv2.imread("IMG20240328155110.jpg") # Read image
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

colors_hsv = {
  "red": {
    "lower": np.array([156,180,100]),
    "upper": np.array([180,255,255]) 
  },
  "yellow": {
    "lower": np.array([26,120,150]),
    "upper": np.array([34,255,255]) 
  },
  "pink": {
    "lower": np.array([156,80,100]),
    "upper": np.array([180,140,255]) 
  }
}

colors_rgb = {
  "red": {
    "lower": np.array([150,5,5]),
    "upper": np.array([220,80,80]) 
  },
  "yellow": {
    "lower": np.array([150,150,50]),
    "upper": np.array([230,230,80]) 
  },
  "pink": {
    "lower": np.array([156,180,100]),
    "upper": np.array([180,255,255]) 
  }
}
for key in colors_hsv.keys():
    lower = colors_hsv[key]["lower"]
    upper = colors_hsv[key]["upper"]
    mask = cv2.inRange(hsv, lower, upper)
    cv2.namedWindow(key, cv2.WINDOW_NORMAL) 
    cv2.imshow(key,mask)


cv2.waitKey(0) # 0==wait forever
cv2.destroyAllWindows() 
