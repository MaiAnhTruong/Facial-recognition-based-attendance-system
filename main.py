# main.py

import os
import cv2
import datetime
import csv
import torch
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
from face_detection import detect_faces, crop_face
from face_encoder import FaceEncoder, encode_face
from face_compare import recognize_face

folder_path = 'E:/Attendance System/image_face'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = FaceEncoder().to(device)
encoder.eval()

def load_known_faces(folder_path, encoder, device):
    known_face_encodings = {}
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        
        encodings = []
        for filename in os.listdir(person_folder):
            image_path = os.path.join(person_folder, filename)
            image = cv2.imread(image_path)
            boxes = detect_faces(image)
            if boxes:
                face = crop_face(image, boxes[0])
                face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                encoding = encode_face(face_image, encoder, device)
                encodings.append(encoding)
        
        if encodings:
            known_face_encodings[person_name] = encodings
    return known_face_encodings

known_face_encodings = load_known_faces(folder_path, encoder, device)

def save_to_csv(name, detection_time):
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, detection_time])

def save_recognized_image(name, frame, box):
    person_folder = os.path.join(folder_path, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(person_folder, f"{name}_{timestamp}.jpg")
    
    top, right, bottom, left = box
    face_image = frame[top:bottom, left:right]
    cv2.imwrite(image_path, face_image)

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System")
        self.root.geometry("800x600")
        
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.start_button = tk.Button(root, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack()

        self.cap = cv2.VideoCapture(0)
        self.is_recognizing = False

        self.update_video_stream()

    def update_video_stream(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.is_recognizing:
            self.recognize_face(frame)

        self.root.after(10, self.update_video_stream)

    def start_recognition(self):
        if not self.is_recognizing:
            self.is_recognizing = True

    def recognize_face(self, frame):
        boxes = detect_faces(frame)
        if boxes:
            face = crop_face(frame, boxes[0])
            face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_encoding = encode_face(face_image, encoder, device)
            name = recognize_face(known_face_encodings, face_encoding)

            if name:
                detection_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_to_csv(name, detection_time)
                save_recognized_image(name, frame, boxes[0])
                
                x, y, width, height = boxes[0]
                cv2.rectangle(frame, (y, x), (y + width, x + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} recognized", (y, x - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                messagebox.showinfo("Recognition Result", f"{name} recognized successfully")
                self.is_recognizing = False
            else:
                messagebox.showinfo("Recognition Result", "Registered user not recognized")
                self.is_recognizing = False

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

root = tk.Tk()
app = AttendanceApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()

