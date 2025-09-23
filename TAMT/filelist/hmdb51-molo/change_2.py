# -*- coding: utf-8 -*-
import os
import re

folder = r"G:\TLU\BigData\Data_time\TAMT\filelist\hmdb51-molo"

# Xóa toàn bộ ký tự nguy hiểm ASCII và full-width Unicode
pattern = r'[\\/:*?"<>|\'&!#;\[\]\(\)\{\}（）]'

for dirname, _, filenames in os.walk(folder):
    for filename in filenames:
        old_path = os.path.join(dirname, filename)
        name, ext = os.path.splitext(filename)
        
        # Xóa ký tự cấm
        new_name = re.sub(pattern, '', name)
        new_path = os.path.join(dirname, new_name + ext)
        
        # Thêm hậu tố nếu trùng tên
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
