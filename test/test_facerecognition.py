import face_recognition
import os

KNOWN_DIR = r"D:\Pycharm_Projects\FaceAttendance_System\test_image\Known_People"
UNKNOWN_DIR = r"D:\Pycharm_Projects\FaceAttendance_System\test_image\Unknown_People"

# Bước 1: Load và encode các khuôn mặt đã biết
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_DIR):
    path = os.path.join(KNOWN_DIR, filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])
    else:
        print(f"[!] Không tìm thấy khuôn mặt trong {filename}")

# Bước 2: Duyệt ảnh chưa biết
for filename in os.listdir(UNKNOWN_DIR):
    path = os.path.join(UNKNOWN_DIR, filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print(f"[?] Không phát hiện khuôn mặt trong {filename}")
        continue

    unknown_encoding = encodings[0]
    results = face_recognition.compare_faces(known_encodings, unknown_encoding)

    print(f"\n📷 Ảnh: {filename}")
    if True in results:
        matched_index = results.index(True)
        print(f"✅ Trùng khớp với: {known_names[matched_index]}")
    else:
        print("❌ Không nhận diện được người trong ảnh.")