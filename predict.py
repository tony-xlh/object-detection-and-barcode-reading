from ultralytics import YOLO
import cv2
from dbr import *

model = YOLO("yolov8n.pt")
error = BarcodeReader.init_license("t0068lQAAALii1mZiwN7g2vIt2Aec77G3J6sLeDho9pOkvCisWjwtsCDVOzSw8uqsllPX2YBOw+Ug6U+yp4zGwY4sn2PK4mY=")
if error[0] != EnumErrorCode.DBR_OK:
   # Add your code for license error processing
   print("License error: "+ error[1])
reader = BarcodeReader()
settings = reader.get_runtime_settings()
settings.barcode_format_ids = EnumBarcodeFormat.BF_DATAMATRIX
reader.update_runtime_settings(settings)

def detect(img):
    results = model.predict(source=img)  # save predictions as labels
    boxes = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0]
            box_to_append = {}
            box_to_append["x"] = int(xyxy[0])
            box_to_append["y"] = int(xyxy[1])
            box_to_append["w"] = int(xyxy[2]) - int(xyxy[0])
            box_to_append["h"] = int(xyxy[3]) - int(xyxy[1])
            box_to_append["conf"] = float(box.conf)
            class_index = int(box.cls[0])
            box_to_append["class"] = model.names[class_index]
            boxes.append(box_to_append)
    return boxes

def draw_results(img,boxes):
    target_width = 640
    target_height = int(640*(img.shape[0]/img.shape[1]))
    ratio = target_width / img.shape[1]
    resized = cv2.resize(img, dsize=(target_width, target_height))
    for box in boxes:
        has_barcodes = True
        box_image = img[box["y"]:box["y"]+box["h"],box["x"]:box["x"]+box["w"]]
        try:
            barcode_results = reader.decode_buffer(box_image)
            if barcode_results == None:
                has_barcodes = False
        except BarcodeReaderError as bre:
            print(bre)
        x = int(ratio*box["x"])
        y = int(ratio*box["y"])
        w = int(ratio*box["w"])
        h = int(ratio*box["h"])
        color = (255,0,0)
        if has_barcodes == False:
            color = (0,0,255)
        cv2.rectangle(resized, (x, y), (x+w, y+h), color=color, thickness=2)
        cv2.putText(resized, box["class"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL) 
    cv2.imshow("result", resized)
    cv2.waitKey(0) # 0==wait forever
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    img = cv2.imread("IMG20240326163255.jpg")
    boxes = detect(img)
    draw_results(img,boxes)

