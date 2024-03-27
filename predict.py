from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

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
        x = int(ratio*box["x"])
        y = int(ratio*box["y"])
        w = int(ratio*box["w"])
        h = int(ratio*box["h"])
        cv2.rectangle(resized, (x, y), (x+w, y+h), color=(255,0,0), thickness=2)
        cv2.putText(resized, box["class"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL) 
    cv2.imshow("result", resized)
    cv2.waitKey(0) # 0==wait forever
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    img = cv2.imread("IMG20240326163255.jpg")
    boxes = detect(img)
    draw_results(img,boxes)

