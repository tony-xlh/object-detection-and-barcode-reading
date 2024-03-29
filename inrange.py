import cv2
import numpy as np

colors_hsv = {
  "red": {
    "lower": np.array([156,150,100]),
    "upper": np.array([180,255,255]) 
  },
  "yellow": {
    "lower": np.array([26,120,150]),
    "upper": np.array([34,255,255]) 
  },
  "pink": {
    "lower": np.array([156,80,100]),
    "upper": np.array([180,140,255]) 
  },
  "green": {
    "lower": np.array([156,80,100]),
    "upper": np.array([200,140,255]) 
  },
  "blue": {
    "lower": np.array([100,80,100]),
    "upper": np.array([120,140,255]) 
  }
}

def get_bounding_rects(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    height, width = img.shape
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w > 80 and h > 80 and w < width/2 and w>4*h:
            rects.append(rect)
    return rects
    
def draw_rects(rects,original_image):
    height, width, _ = original_image.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    for (x,y,w,h) in rects:
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        cv2.rectangle(original_image, (x,y), (x+w,y+h), (255, 0, 0), 5)

def detect(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    key = "red"
    lower = colors_hsv[key]["lower"]
    upper = colors_hsv[key]["upper"]
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    #thresh = cv2.dilate(thresh, kernel)
    thresh = cv2.erode(thresh, kernel)
    rects1 = get_bounding_rects(thresh)
    rects2 = get_bounding_rects(mask)
    rects = rects1 + rects2
    return rects
    
if __name__ == "__main__":
    image = cv2.imread("IMG20240328155110.jpg")
    rects = detect(image)
    draw_rects(rects,image)
    print("Found " + str(len(rects)) + " rectangle(s)")
    cv2.imwrite("image.jpg",image)

