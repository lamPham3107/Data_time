import os
import json
import sys

# Fix Unicode cho Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Thư mục chứa file JSON
base_dir = "g:/TLU/BigData/Data_time/TAMT/filelist/hmdb51-molo"

# File gốc và file output
json_path = os.path.join(base_dir, "val_old.json")
output_path = os.path.join(base_dir, "val.json")

# Prefix mới cho đường dẫn video
new_prefix = "G:/TLU/BigData/data_down/hmdb51_org/"

# Đọc file JSON gốc
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Cập nhật đường dẫn mới
new_image_names = []
for p in data["image_names"]:
    filename = os.path.basename(p)      # Lấy tên file .avi
    class_name = p.split("/")[-2]       # Lấy tên class, vd brush_hair
    new_path = os.path.join(new_prefix, class_name, filename)
    new_image_names.append(new_path.replace("\\", "/"))

data["image_names"] = new_image_names

# Lưu file JSON mới vào cùng thư mục
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f" Save success: {output_path}")
