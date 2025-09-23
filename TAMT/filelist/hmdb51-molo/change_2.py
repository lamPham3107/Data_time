# -*- coding: utf-8 -*-
import os
import re
folder = r"G:\TLU\BigData\data_down\hmdb51_org"

for dirname, _, filenames in os.walk(folder):
    for filename in filenames:
        old_path = os.path.join(dirname, filename)
        new_name = re.sub(r'[\[\];#!&]', '', filename)  # Bỏ ký tự cấm
        new_path = os.path.join(dirname, new_name)
        
        if old_path == new_path:
            continue  # Không đổi nếu giống nhau

        if os.path.exists(new_path):
            print(f" Bo qua vi trung: {new_path}")
            continue  # Bỏ qua nếu file mới đã tồn tại

        try:
            os.rename(old_path, new_path)
            print(f"Đoi ten: {old_path} -> {new_path}")
        except Exception as e:
            print(f"Loi {old_path}: {e}")

