# --- PHẦN 1: THIẾT LẬP (Bao gồm Albumentations) ---
import mediapipe as mp
import cv2
import os
import json
import pandas as pd
import numpy as np
import albumentations as A  # Thư viện augmentation

# 1.1. Chỉ nhập các giải pháp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 1.2. Định nghĩa "công thức" Augmentation
# (Công thức này sẽ CHỈ áp dụng cho tập Train)
augment_pipeline = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

# 1.3. Số lượng "bản sao ảo" muốn tạo cho MỖI ảnh
N_AUGMENTATIONS_PER_IMAGE = 4 

print("Đã nhập các thư viện và định nghĩa Augmentation Pipeline.")

# --- PHẦN 2: ĐỊNH NGHĨA HÀM XỬ LÝ ---
# Thêm cờ 'is_training' để biết khi nào cần augment
def process_dataset(json_path, image_dir, is_training=False):
    
    print(f"\n[Bắt đầu] Xử lý bộ dữ liệu tại: {json_path}")
    # In ra trạng thái Augment
    print(f"  -> Augmentation: {'BẬT' if is_training else 'TẮT'}")

    # Tạo một "cỗ máy" MỚI cho mỗi bộ (train/valid/test)
    hands_processor = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    print("  -> Đã khởi tạo cỗ máy MediaPipe MỚI.")
    
    # Mở và đọc file COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Tạo bản đồ tra cứu (giữ nguyên)
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    category_id_to_label_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    image_filename_to_label = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        filename = image_id_to_filename[image_id]
        label = category_id_to_label_name[category_id]
        image_filename_to_label[filename] = label

    processed_data_list = []
    total_images = len(image_filename_to_label)
    print(f"  -> Tìm thấy {total_images} ảnh gốc để xử lý.")

    for i, (filename, label) in enumerate(image_filename_to_label.items()):
        
        if (i + 1) % 500 == 0: 
            print(f"    ...Đã xử lý {i+1} / {total_images} ảnh gốc...")

        # Đọc ảnh (Dùng np.fromfile)
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            continue
        try:
            n = np.fromfile(img_path, np.uint8)
            image = cv2.imdecode(n, cv2.IMREAD_COLOR)
        except Exception:
            continue
        if image is None:
             continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Tạo danh sách ảnh cần xử lý
        images_to_process = []
        images_to_process.append(image_rgb) # 1. Luôn thêm ảnh GỐC
        
        # 2. CHỈ KHI là tập train, tạo thêm các ảnh "ảo"
        if is_training:
            for _ in range(N_AUGMENTATIONS_PER_IMAGE):
                augmented = augment_pipeline(image=image_rgb)
                images_to_process.append(augmented['image'])
        
        # Lặp qua (1 ảnh gốc) hoặc (N+1 ảnh)
        for img_to_process in images_to_process:
            results = hands_processor.process(img_to_process)
            if results.multi_hand_landmarks:
                # (Code trích xuất 63 điểm giữ nguyên)
                hand_landmarks = results.multi_hand_landmarks[0]
                wrist_landmark = hand_landmarks.landmark[0]
                relative_landmarks_flat = []
                for landmark in hand_landmarks.landmark:
                    relative_landmarks_flat.append(landmark.x - wrist_landmark.x)
                    relative_landmarks_flat.append(landmark.y - wrist_landmark.y)
                    relative_landmarks_flat.append(landmark.z - wrist_landmark.z)
                row = [label] + relative_landmarks_flat
                processed_data_list.append(row)
            else:
                pass # Bỏ qua trong im lặng

    print(f"[Hoàn thành] Trích xuất được {len(processed_data_list)} mẫu hợp lệ.")
    hands_processor.close()
    return processed_data_list

# --- PHẦN 3: ĐỊNH NGHĨA ĐƯỜNG DẪN & CHẠY  ---
print(f"\nThư mục làm việc hiện tại: {os.getcwd()}")
base_data_directory = os.path.join("1_data", "raw", "data_mini_app_01")
print(f"Đường dẫn dữ liệu gốc: {base_data_directory}") 

# Đường dẫn Train
train_json_path = os.path.join(base_data_directory, "train", "_annotations.coco.json")
train_image_directory = os.path.join(base_data_directory, "train")
# Đường dẫn Valid
valid_json_path = os.path.join(base_data_directory, "valid", "_annotations.coco.json")
valid_image_directory = os.path.join(base_data_directory, "valid")
# SỬA: Thêm đường dẫn Test
test_json_path = os.path.join(base_data_directory, "test", "_annotations.coco.json")
test_image_directory = os.path.join(base_data_directory, "test")

# Kiểm tra đường dẫn
if not os.path.exists(train_json_path):
    print(f"\n--- LỖI! KHÔNG TÌM THẤY FILE JSON (Train) ---")
    exit()
else:
    print(f"OK! Đã tìm thấy file JSON train.")

# Chạy xử lý
# is_training=True -> Bật Augmentation
train_data_rows = process_dataset(train_json_path, train_image_directory, is_training=True)
# is_training=False -> Tắt Augmentation
valid_data_rows = process_dataset(valid_json_path, valid_image_directory, is_training=False)
# SỬA: Chạy xử lý cho tập Test (Tắt Augmentation)
test_data_rows = process_dataset(test_json_path, test_image_directory, is_training=False)


# --- PHẦN 4: LƯU KẾT QUẢ RA FILE CSV (SỬA: Thêm TEST) ---
columns = ['label'] 
for i in range(21):
    columns += [f'x{i}', f'y{i}', f'z{i}']
print(f"\nChuẩn bị lưu file CSV với {len(columns)} cột.")

df_train = pd.DataFrame(train_data_rows, columns=columns)
df_valid = pd.DataFrame(valid_data_rows, columns=columns)
# SỬA: Thêm DataFrame cho Test
df_test = pd.DataFrame(test_data_rows, columns=columns)

output_dir = os.path.join("1_data", "processed")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục: {output_dir}")

train_csv_filename = os.path.join(output_dir, "train_landmarks_augmented.csv") 
valid_csv_filename = os.path.join(output_dir, "valid_landmarks.csv")
# SỬA: Thêm file cho Test
test_csv_filename = os.path.join(output_dir, "test_landmarks.csv")

df_train.to_csv(train_csv_filename, index=False)
df_valid.to_csv(valid_csv_filename, index=False)
df_test.to_csv(test_csv_filename, index=False) # SỬA: Lưu file test

# --- PHẦN 5: DỌN DẸP VÀ HOÀN TẤT (SỬA: Thêm TEST) ---
print("\n--- TẤT CẢ HOÀN TẤT! ---")
print(f"Đã lưu dữ liệu train vào file: {train_csv_filename} ({len(df_train)} hàng)")
print(f"Đã lưu dữ liệu valid vào file: {valid_csv_filename} ({len(df_valid)} hàng)")
print(f"Đã lưu dữ liệu test vào file: {test_csv_filename} ({len(df_test)} hàng)")