import os, re

folder = r"G:\TLU\BigData\Data_time\TAMT\filelist\hmdb51-molo"

def clean_filename(filename):
    # Giữ lại chữ cái, số, ., -, _
    return re.sub(r'[^A-Za-z0-9._-]', '', filename)

for dirname, _, filenames in os.walk(folder):
    for filename in filenames:
        old_path = os.path.join(dirname, filename)
        new_name = clean_filename(filename)
        new_path = os.path.join(dirname, new_name)

        if old_path == new_path:
            continue
        if os.path.exists(new_path):
            print(f" Bo qua vi trung: {new_path}")
            continue

        try:
            os.rename(old_path, new_path)
            print(f" Đoi ten: {old_path} -> {new_path}")
        except Exception as e:
            print(f" Loi {old_path}: {e}")
