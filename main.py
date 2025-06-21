import os
import datetime
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import util
import numpy as np
import face_recognition
import csv
from tkinter.ttk import Combobox



class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'Register new user', 'gray', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=300)

        self.view_attendance_button = util.get_button(
            self.main_window, 'View Attendance', 'green', self.view_attendance)
        self.view_attendance_button.place(x=750, y=400)

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
        known_ids = []

        for file in os.listdir(self.db_dir):
            if file.endswith('.jpg'):
                img_path = os.path.join(self.db_dir, file)
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    student_id = os.path.splitext(file)[0]
                    known_ids.append(student_id)

        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.45)
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

        if True in matches:
            best_match_idx = np.argmin(face_distances)
            student_id = known_ids[best_match_idx]

            # 🔎 Lấy thông tin từ students.csv
            full_name = ""
            class_name = ""
            try:
                with open("students.csv", newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['student_id'] == student_id:
                            full_name = row['full_name']
                            class_name = row['class']
                            break
            except Exception as e:
                util.msg_box("Error", f"Failed to read students.csv:\n{e}")
                os.remove(unknown_img_path)
                return

            # ⏱ Thời gian hiện tại
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 📁 Ghi file điểm danh
            try:
                already_logged = False
                if os.path.exists("attendance.csv"):
                    with open("attendance.csv", newline='', encoding='utf-8') as f:
                        for line in f:
                            if student_id in line and timestamp.split()[0] in line:
                                already_logged = True
                                break

                if not already_logged:
                    with open("attendance.csv", "a", encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        if os.stat("attendance.csv").st_size == 0:
                            writer.writerow(['student_id', 'full_name', 'class', 'timestamp'])  # Header
                        writer.writerow([student_id, full_name, class_name, timestamp])
                    util.msg_box('Welcome back!', f'Welcome, {full_name}.')
                else:
                    util.msg_box('Info', f'{full_name} has already been marked present today.')

            except Exception as e:
                util.msg_box("Error", f"Cannot write to attendance.csv:\n{e}")
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

        # ✅ Load danh sách MSSV chưa đăng ký
        student_ids = []
        try:
            with open("students.csv", newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    student_id = row['student_id']
                    if f"{student_id}.jpg" not in os.listdir(self.db_dir):
                        student_ids.append(student_id)
        except Exception as e:
            util.msg_box("Error", f"Cannot load students.csv:\n{e}")
            return

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Select Student ID:')
        self.text_label_register_new_user.place(x=750, y=70)

        self.combo_ids = Combobox(self.register_new_user_window, values=student_ids, state='readonly')
        self.combo_ids.place(x=750, y=150)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_pil.copy()

    def start(self):
        self.main_window.mainloop()

    def view_attendance(self):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        records = []

        try:
            with open("attendance.csv", newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['timestamp'].startswith(today):
                        records.append(row)
        except FileNotFoundError:
            util.msg_box("Info", "No attendance records found.")
            return

        if not records:
            util.msg_box("Info", f"No one has been marked present today ({today}).")
            return

        # Hiển thị danh sách trong cửa sổ mới
        win = tk.Toplevel(self.main_window)
        win.title(f"Attendance for {today}")
        win.geometry("600x400+400+200")

        header = tk.Label(win, text="Today's Attendance", font=("Arial", 16, "bold"))
        header.pack(pady=10)

        text = tk.Text(win, font=("Consolas", 11))
        text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Ghi tiêu đề
        text.insert(tk.END, f"{'ID':<12}{'Name':<30}{'Class':<15}{'Time':<20}\n")
        text.insert(tk.END, "-" * 80 + "\n")

        for row in records:
            text.insert(
                tk.END,
                f"{row['student_id']:<12}{row['full_name']:<30}{row['class']:<15}{row['timestamp'].split()[1]:<20}\n"
            )

        text.config(state=tk.DISABLED)

    def accept_register_new_user(self):
        name = self.combo_ids.get().strip()

        if not name:
            util.msg_box('Error', 'Please select a student ID.')
            return

        if f'{name}.jpg' in os.listdir(self.db_dir):
            util.msg_box('Warning', f'Student ID "{name}" already registered.')
            return

        img_array = np.array(self.register_new_user_capture)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        filename = os.path.join(self.db_dir, f'{name}.jpg')
        cv2.imwrite(filename, img_bgr)

        util.msg_box('Success !', f'Student {name} registered successfully!')
        self.register_new_user_window.destroy()

        img_array = np.array(self.register_new_user_capture)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        filename = os.path.join(self.db_dir, f'{name}.jpg')
        cv2.imwrite(filename, img_bgr)

        util.msg_box('Success !', 'User was registered successfully!')
        self.register_new_user_window.destroy()


if __name__ == '__main__':
    app = App()
    app.start()
