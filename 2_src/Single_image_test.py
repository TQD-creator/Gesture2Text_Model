import mediapipe as mp
import cv2
import os

# --- 1. KHỞI TẠO MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_processor = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3  # Giữ ở mức 0.3
)
print("Đã khởi tạo MediaPipe...")

# --- 2. TẢI 1 ẢNH DUY NHẤT ---

# !!! CHỈNH LẠI ĐƯỜNG DẪN NÀY CHO ĐÚNG !!!
image_path = r"1_data\raw\data_mini_app_01\valid\Y_187_jpg.rf.c6d0a299ee3c17fd7c393e6a84e6037a.jpg"

# Kiểm tra xem file có tồn tại không
if not os.path.exists(image_path):
    print(f"--- LỖI ---")
    print(f"Không tìm thấy file ảnh tại: {image_path}")
    print("Vui lòng kiểm tra lại đường dẫn và tên file ảnh!")
    exit()

print(f"Đang tải ảnh: {image_path}")
image = cv2.imread(image_path)

if image is None:
    print("--- LỖI ---")
    print("cv2.imread() trả về 'None'. File ảnh có thể bị hỏng.")
    exit()

print("Tải ảnh thành công. Bắt đầu chuyển hệ màu...")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Đã chuyển sang RGB. Bắt đầu xử lý MediaPipe...")

# --- 3. CHẠY MEDIAPIPE ---
results = hands_processor.process(image_rgb)

# --- 4. BÁO CÁO KẾT QUẢ ---
if results.multi_hand_landmarks:
    print("\n--- THÀNH CÔNG! ---")
    print("Đã tìm thấy tay trong ảnh.")
    
    # Vẽ các điểm tìm thấy lên ảnh GỐC (để kiểm tra)
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)
    
    # Lưu ảnh kết quả ra
    output_filename = "output_debug_image.jpg"
    cv2.imwrite(output_filename, annotated_image)
    print(f"Đã lưu ảnh kết quả vào: {output_filename}")
    print("Hãy mở file này lên xem đã vẽ đúng chưa!")

else:
    print("\n--- THẤT BẠI! ---")
    print("Không tìm thấy bất kỳ bàn tay nào trong ảnh.")

# Dọn dẹp
hands_processor.close()
print("Đã đóng MediaPipe.")