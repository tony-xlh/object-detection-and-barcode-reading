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
    "lower": np.array([88,50,100]),
    "upper": np.array([99,80,255]) 
  },
  "blue": {
    "lower": np.array([78,120,100]),
    "upper": np.array([99,255,255]) 
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
    rects = []
    for key in colors_hsv:
        lower = colors_hsv[key]["lower"]
        upper = colors_hsv[key]["upper"]
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        #thresh = cv2.dilate(thresh, kernel)
        mask = cv2.erode(mask, kernel)
        cv2.imwrite("mask-"+key+".jpg",mask)
        rects_of_one_color = get_bounding_rects(mask)
        rects = rects + rects_of_one_color
    rects = filter_out_small_areas(rects)
    return rects
    
    
def filter_out_small_areas(rects):
    new_rects = []
    for i in range(0,len(rects)):
        rect1 = rects[i]
        add = True
        for j in range(0,len(rects)):
            if j == i:
                continue
            rect2 = rects[j]
            area = get_overlap_area(rect1,rect2)
            print(area)
            if area>0:
                area1 = rect1[2]*rect1[3]
                area2 = rect2[2]*rect2[3]
                min_area = min(area1,area2)
                percent = area/min_area
                print(percent)
                if  percent>0.8 and area1<area2:
                    print("do not add")
                    add = False
                    break
        if add == True:
            new_rects.append(rect1)
    return new_rects
    
def get_overlap_area(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    right1 = x1+w1
    right2 = x2+w2
    bottom1 = y1+h1
    bottom2 = y2+h2
    
    x_overlap = max(0, min(right1, right2) - max(x1,x2))

    y_overlap = max(0, min(bottom1, bottom2) - max(y1,y2))

    overlap_area = x_overlap * y_overlap
    return overlap_area
    
if __name__ == "__main__":
    image = cv2.imread("./samples/IMG20240401165353.jpg")
    rects = detect(image)
    draw_rects(rects,image)
    print("Found " + str(len(rects)) + " rectangle(s)")
    cv2.imwrite("image.jpg",image)

