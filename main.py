import os
import datetime
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import util
import numpy as np
import face_recognition


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'Register new user', 'gray', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)
    
    def login(self):
        unknown_img_path = './.tmp.jpg'
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)

        unknown_img = face_recognition.load_image_file(unknown_img_path)
        unknown_encs = face_recognition.face_encodings(unknown_img)

        if not unknown_encs:
            util.msg_box("Error", "No face found in captured image.")
            os.remove(unknown_img_path)
            return

        unknown_encoding = unknown_encs[0]
        known_encodings = []
        known_names = []

        for file in os.listdir(self.db_dir):
            if file.endswith('.jpg'):
                img_path = os.path.join(self.db_dir, file)
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    name = os.path.splitext(file)[0]  # ✅ không dùng split('_')
                    known_names.append(name)

        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.45)
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

        if True in matches:
            best_match_idx = np.argmin(face_distances)
            name = known_names[best_match_idx]
            util.msg_box('Welcome back!', f'Welcome, {name}.')
            with open(self.log_path, 'a') as f:
                f.write('{}, login at {}\n'.format(name, datetime.datetime.now()))
        else:
            util.msg_box('Oh no ....', 'Unknown person. Please try again or register new user.')

        os.remove(unknown_img_path)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please, input username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_pil.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()

        if not name or not name.isalnum():
            util.msg_box('Error', 'Username must be alphanumeric and non-empty.')
            return

        if f'{name}.jpg' in os.listdir(self.db_dir):
            util.msg_box('Warning', f'User name "{name}" already exists. Please choose a different name.')
            return

        img_array = np.array(self.register_new_user_capture)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        filename = os.path.join(self.db_dir, f'{name}.jpg')
        cv2.imwrite(filename, img_bgr)

        util.msg_box('Success !', 'User was registered successfully!')
        self.register_new_user_window.destroy()


if __name__ == '__main__':
    app = App()
    app.start()
