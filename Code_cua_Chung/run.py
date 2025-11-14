import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ==============================
# 🔹 1. Load model và scaler
# ==============================
MODEL_PATH = 'model_mlp.pkl'
SCALER_PATH = 'scaler.pkl'

try:
    print("🔄 Đang tải model và scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Đã tải model và scaler!")
except Exception as e:
    print(f"❌ Không thể tải model hoặc scaler: {e}")
    print("➡️ Vui lòng kiểm tra lại phiên bản thư viện (numpy/scikit-learn) hoặc huấn luyện lại model.")
    exit()

# ==============================
# 🔹 2. Khởi tạo Mediapipe
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Cho phép nhận diện cả 2 tay (nhưng logic bên dưới chỉ xử lý 1 tay)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ==============================
# 🔹 3. Biến điều khiển
# ==============================
sentence_raw = ""
last_detection_time = time.time()
last_recognition_time = 0
running = True
cap = None

# ==============================
# 🔹 4. Hàm xử lý Reset & Thoát
# ==============================
def reset_text():
    global sentence_raw
    sentence_raw = ""
    label_text.set("")
    print("\n--- KẾT QUẢ ĐÃ ĐƯỢC RESET ---")

def quit_app():
    global running
    running = False
    # Đợi thread camera dừng hẳn
    time.sleep(0.5) 
    if cap:
        cap.release()
    root.destroy()

# ==============================
# 🔹 5. Hàm xử lý Camera
# ==============================
def camera_loop():
    global last_detection_time, last_recognition_time, sentence_raw, cap

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở camera.")
        return

    while running:
        ret, frame = cap.read()
        if not ret or not running:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        current_time = time.time()

        if results.multi_hand_landmarks:
            # Chỉ xử lý 1 tay (tay đầu tiên phát hiện được)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Vẽ skeleton lên frame (cho phần hiển thị bên trái)
            drawing_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) 
            mp_drawing.draw_landmarks(
                drawing_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # === 💡 [ĐÂY LÀ PHẦN SỬA LỖI] ===
            # Chuyển đổi toạ độ TUYỆT ĐỐI (từ camera)
            # sang toạ độ TƯƠNG ĐỐI (so với cổ tay)
            # để khớp với dữ liệu training (train.csv)
            
            all_landmarks_list = hand_landmarks.landmark
            
            # 2. Lấy toạ độ gốc (cổ tay - điểm 0)
            base_x, base_y, base_z = all_landmarks_list[0].x, all_landmarks_list[0].y, all_landmarks_list[0].z

            landmarks_relative = []
            
            # 3. Tính toạ độ tương đối của TẤT CẢ 21 điểm
            for lm in all_landmarks_list:
                landmarks_relative.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            # ==================================

            # Logic quét 1 giây 1 lần
            if current_time - last_recognition_time >= 1.0:
                
                # 4. Dùng 'landmarks_relative' (63 features) để dự đoán
                X_input = np.array(landmarks_relative).reshape(1, -1)
                
                X_scaled = scaler.transform(X_input)
                y_pred = model.predict(X_scaled)
                detected_letter = y_pred[0]
                last_recognition_time = current_time

                sentence_raw += detected_letter
                label_text.set(sentence_raw)
                
                print(f"Câu thô (raw): {sentence_raw}")

            last_detection_time = time.time()
            
            # Dùng frame đã vẽ skeleton
            final_frame_for_gui = drawing_frame 

        else:
            # Không phát hiện tay
            final_frame_for_gui = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) # Dùng frame gốc
            
            # Logic thêm dấu cách sau 1.5s
            if current_time - last_detection_time > 1.5:
                if len(sentence_raw) > 0 and not sentence_raw.endswith(" "):
                    sentence_raw += " "
                    label_text.set(sentence_raw)
                    
                    print(f"Câu thô (raw): {sentence_raw}")
                    
                last_detection_time = current_time 

        # Convert ảnh cho Tkinter hiển thị
        try:
            img = Image.fromarray(cv2.cvtColor(final_frame_for_gui, cv2.COLOR_BGR2RGB))
            img = img.resize((640, 480)) 
            imgtk = ImageTk.PhotoImage(image=img)

            # Hiển thị lên GUI
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        except Exception as e:
            # Bỏ qua lỗi nếu GUI đã bị đóng
            if running:
                print(f"Lỗi cập nhật GUI: {e}")


    if cap:
        cap.release()

# ==============================
# 🔹 6. Giao diện Tkinter
# ==============================
root = tk.Tk()
root.title("Vietnamese Sign Language Recognition (Model 1)")

root.geometry("1280x480")
root.resizable(False, False)

# Khung trái (camera) - 640x480
frame_left = tk.Frame(root, width=640, height=480, bg="black")
frame_left.pack(side="left", fill="both", expand=True)
frame_left.pack_propagate(False) 

video_label = tk.Label(frame_left, bg="black")
video_label.pack(fill="both", expand=True)

# Khung phải (text + nút) - 640x480
frame_right = tk.Frame(root, width=640, height=480, bg="#1E1E1E")
frame_right.pack(side="right", fill="both", expand=True)
frame_right.pack_propagate(False) 

# Label kết quả
label_title = tk.Label(frame_right, text="Kết quả nhận diện", font=("Arial", 18, "bold"), fg="white", bg="#1E1E1E")
label_title.pack(pady=(20, 10)) 

text_display_frame = tk.Frame(frame_right, bg="#1E1E1E", height=300, width=600)
text_display_frame.pack(pady=10)
text_display_frame.pack_propagate(False)

label_text = tk.StringVar()
label_display = tk.Label(text_display_frame, textvariable=label_text, font=("Consolas", 20), fg="#00FF00", bg="#1E1E1E", wraplength=580, justify="left", anchor="nw")
label_display.pack(fill="both", expand=True, padx=10)

# Nút Reset và Thoát
btn_frame = tk.Frame(frame_right, bg="#1E1E1E")
btn_frame.pack(pady=20) 

btn_reset = tk.Button(btn_frame, text="🔁 Reset", command=reset_text, width=10, height=2, bg="#007ACC", fg="white", font=("Arial", 12, "bold"), relief="raised", borderwidth=2)
btn_reset.pack(side="left", padx=20)

btn_quit = tk.Button(btn_frame, text="❌ Thoát", command=quit_app, width=10, height=2, bg="#D9534F", fg="white", font=("Arial", 12, "bold"), relief="raised", borderwidth=2)
btn_quit.pack(side="right", padx=20)

# Xử lý nút X (WM_DELETE_WINDOW)
root.protocol("WM_DELETE_WINDOW", quit_app)

# Hỏi bật camera
if messagebox.askyesno("Bật camera", "📷 Bạn có cho phép mở camera để nhận diện tay không?"):
    threading.Thread(target=camera_loop, daemon=True).start()
else:
    messagebox.showinfo("Thoát", "Bạn đã từ chối bật camera. Ứng dụng sẽ đóng.")
    root.destroy()

# Vòng lặp GUI
if 'root' in locals() and root.winfo_exists():
    root.mainloop()

print("Ứng dụng đã đóng.")