# object-detection-and-barcode-reading

A Python demo running object detection and barcode reading.

It first detects objects using Yolo v8 and then read barcodes from the detected objects with [Dynamsoft Barcode Reader](https://www.dynamsoft.com/barcode-reader/overview/).

The objects will be drawn on the image. If the object does not contain a barcode, it will be highlighted.

![image](https://github.com/tony-xlh/object-detection-and-barcode-reading/assets/112376616/9a34411a-53a3-4a03-af74-8f34a528f264)


Files explaination:

* `predict.py`: use YOLOv8 to detect objects and read barcodes.
* `opencv.py`: a version using OpenCV's DNN for inference.
* `convertToONNX.py`: convert the YOLO model into ONNX.
