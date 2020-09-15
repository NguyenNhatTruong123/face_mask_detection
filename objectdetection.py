from imageai.Detection import ObjectDetection
import os
import pandas as pd

def objectdetec(img):
    #img='test'
    img_path=img+'.png'
    # img_dec=img+str(i)+'new.png'
    execution_path = os.getcwd()
    img_dec=img+'new.png'

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , img_path), output_image_path=os.path.join(execution_path ,img_dec ))

    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"])
    
#objectdetec('test11')






