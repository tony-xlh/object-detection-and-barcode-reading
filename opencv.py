import cv2.dnn
import numpy as np
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
from dbr import *

CLASSES = yaml_load(check_yaml("coco128.yaml"))["names"]

model = cv2.dnn.readNetFromONNX("yolov8n.onnx")
error = BarcodeReader.init_license("t0068lQAAALii1mZiwN7g2vIt2Aec77G3J6sLeDho9pOkvCisWjwtsCDVOzSw8uqsllPX2YBOw+Ug6U+yp4zGwY4sn2PK4mY=")
if error[0] != EnumErrorCode.DBR_OK:
   # Add your code for license error processing
   print("License error: "+ error[1])
reader = BarcodeReader()
settings = reader.get_runtime_settings()
settings.barcode_format_ids = EnumBarcodeFormat.BF_DATAMATRIX
reader.update_runtime_settings(settings)

def detect(image):
    ratio_x = 640/image.shape[1]
    ratio_y = 640/image.shape[0]
    # Prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    print("rows")
    print(rows)
    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "x": int(box[0]/ratio_x),
            "y": int(box[1]/ratio_y),
            "w": int(box[2]/ratio_x),
            "h": int(box[3]/ratio_y)
        }
        detections.append(detection)
    
    print(detections)
    print(len(detections))
    return detections

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