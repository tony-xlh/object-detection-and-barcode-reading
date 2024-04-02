import cv2
import shutil
from inrange import detect
import os

def create():
    make_folders()
    make_yaml()
    folder = "cassette"
    filenames = os.listdir(folder)
    num_train = len(filenames)/2
    num_val = len(filenames) - num_train
    file_numbers = 0
    for filename in filenames:
        file_numbers = file_numbers + 1
        path = folder+"/"+filename
        img = cv2.imread(path)
        rects = detect(img,addResultsOfthreshold=True)
        print(len(rects))
        if file_numbers > num_train:
            print("val")
            target_img_folder = "dataset/images/val/"
            target_label_folder = "dataset/labels/val/"
            write_image(img,filename,target_img_folder)
            write_label(rects,img,filename,target_label_folder)
        else:
            print("train")
            target_img_folder = "dataset/images/train/"
            target_label_folder = "dataset/labels/train/"
            write_image(img,filename,target_img_folder)
            write_label(rects,img,filename,target_label_folder)

def write_image(img,image_name,target_folder):
    resized = cv2.resize(img, (640, 640))
    cv2.imwrite(target_folder+image_name,resized)
    

def write_label(rects,img,image_name,target_folder):
    file_name = os.path.splitext(image_name)[0]
    f = open(target_folder+file_name+".txt","w",encoding="utf8")
    height, width, _ = img.shape
    for (x,y,w,h) in rects:
        normalized_x = (x + w/2) / width #center y
        normalized_y = (y + h/2) / height #center x
        normalized_w = w / width
        normalized_h = h / height
        line = f'0 {normalized_x} {normalized_y} {normalized_w} {normalized_h}\n'
        f.write(line)
    f.close()

def make_folders():
    if os.path.exists("dataset") == False:
        os.mkdir("dataset")
    if os.path.exists("dataset/images") == False:
        os.mkdir("dataset/images")
    if os.path.exists("dataset/images/train") == False:
        os.mkdir("dataset/images/train")
    if os.path.exists("dataset/images/val") == False:
        os.mkdir("dataset/images/val")
    if os.path.exists("dataset/labels") == False:
        os.mkdir("dataset/labels")
    if os.path.exists("dataset/labels/train") == False:
        os.mkdir("dataset/labels/train")
    if os.path.exists("dataset/labels/val") == False:
        os.mkdir("dataset/labels/val")

def make_yaml():
    content = '''path: ./ # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Classes
names:
  0: cassette'''
    f = open("dataset/cassette.yaml","w",encoding="utf8")
    f.write(content)
    f.close()

if __name__ == "__main__":
    create()