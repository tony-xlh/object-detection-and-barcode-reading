import cv2.dnn
import numpy as np
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
CLASSES = yaml_load(check_yaml("coco128.yaml"))["names"]

model = cv2.dnn.readNetFromONNX("yolov8n.onnx")

# Read the input image
original_image = cv2.imread("scaled.jpg")
[height, width, _] = original_image.shape

# Prepare a square image for inference
length = max((height, width))
image = np.zeros((length, length, 3), np.uint8)
image[0:height, 0:width] = original_image

cv2.imwrite("out.jpg",image)
# Calculate scale factor
scale = length / 640

# Preprocess the image and prepare blob for model
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
        "class_id": class_ids[index],
        "class_name": CLASSES[class_ids[index]],
        "confidence": scores[index],
        "box": box,
        "scale": scale,
    }
    detections.append(detection)
    
print(detections)
print(len(detections))

