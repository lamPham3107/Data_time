# -*- coding: utf-8 -*-
import os
import re

folder = r"G:\TLU\BigData\Data_time\TAMT\filelist\hmdb51-molo"

# Regex các ký tự cấm
pattern = r'[\\/:*?"<>|\'&!#;\[\]\(\)\{\}]'

for dirname, _, filenames in os.walk(folder):
    for filename in filenames:
        old_path = os.path.join(dirname, filename)
        
        # Tách phần tên và extension
        name, ext = os.path.splitext(filename)
        
        # Xóa tất cả ký tự cấm
        new_name = re.sub(pattern, '', name)
        new_path = os.path.join(dirname, new_name + ext)
        
        # Nếu trùng tên, thêm hậu tố
        count = 1
        while os.path.exists(new_path) and new_path != old_path:
            new_path = os.path.join(dirname, f"{new_name}_{count}{ext}")
            count += 1
        
        if old_path == new_path:
            continue
        
        try:
            os.rename(old_path, new_path)
            print(f"Đổi tên: {old_path} -> {new_path}")
        except Exception as e:
            print(f"Lỗi {old_path}: {e}")
