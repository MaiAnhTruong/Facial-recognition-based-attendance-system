import cv2
from mtcnn import MTCNN

mtcnn = MTCNN()

def detect_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = mtcnn.detect_faces(image_rgb)
    
    bounding_boxes = []
    for face in faces:
        x, y, width, height = face['box']
        bounding_box = (y, x + width, y + height, x)
        bounding_boxes.append(bounding_box)
    
    return bounding_boxes

def crop_face(image, box):
    top, right, bottom, left = box
    return image[top:bottom, left:right]

