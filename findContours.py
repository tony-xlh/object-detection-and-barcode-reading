import cv2
from dbr import *

error = BarcodeReader.init_license("t0068lQAAALii1mZiwN7g2vIt2Aec77G3J6sLeDho9pOkvCisWjwtsCDVOzSw8uqsllPX2YBOw+Ug6U+yp4zGwY4sn2PK4mY=")
if error[0] != EnumErrorCode.DBR_OK:
   # Add your code for license error processing
   print("License error: "+ error[1])
reader = BarcodeReader()
reader.init_runtime_settings_with_file("datamatrix-template.json")

def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    #cv2.imwrite('thresh.jpg', thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w > 50 and h > 20 and w>h:
            rects.append(rect)
    return rects
    
def draw_rects(rects,original_image):
    height, width, _ = original_image.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    for (x,y,w,h) in rects:
        has_barcodes = True
        box_image = img[y:y+h,x:x+w]
        box_image = cv2.resize(box_image,(w*4,h*4))
        try:
            barcode_results = reader.decode_buffer(box_image)
            if barcode_results == None:
                has_barcodes = False
        except BarcodeReaderError as bre:
            print(bre)
        color = (255,0,0)
        if has_barcodes == False:
            color = (0,0,255)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        cv2.rectangle(original_image, (x,y), (x+w,y+h), color, 5)
        
if __name__ == "__main__":
    img = cv2.imread("./samples/barcodes-on-labels.jpg")
    rects = detect(img)
    draw_rects(rects,img)
    cv2.imwrite("image.jpg",img)

