import os
import json

def setup_hdfs_simulation(hdfs_root="/kaggle/working/hdfs_simulation"):
    """Tạo cấu trúc thư mục giống HDFS"""
    os.makedirs(f"{hdfs_root}/input", exist_ok=True)
    os.makedirs(f"{hdfs_root}/map_output", exist_ok=True)
    os.makedirs(f"{hdfs_root}/reduce_output", exist_ok=True)
    print(f"HDFS simulation created at: {hdfs_root}")
    return hdfs_root

def split_dataset_to_hdfs(dataset_path, hdfs_root, num_chunks=4):
    """Chia dataset thành chunks và lưu vào HDFS"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    videos = data['image_names']
    labels = data.get('image_labels', [])
    
    chunk_size = len(videos) // num_chunks
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(videos)
        
        chunk_data = {
            'image_names': videos[start_idx:end_idx],
            'image_labels': labels[start_idx:end_idx] if labels else []
        }
        
        chunk_path = f"{hdfs_root}/input/chunk_{i}.json"
        with open(chunk_path, 'w') as f:
            json.dump(chunk_data, f)
        
        print(f"Created chunk_{i}.json with {len(chunk_data['image_names'])} videos")
    
    return num_chunks

if __name__ == '__main__':
    hdfs_root = setup_hdfs_simulation()
    dataset_path = "/kaggle/input/tamt-bigdata/TAMT/filelist/ucf101-molo/base.json"
    num_chunks = split_dataset_to_hdfs(dataset_path, hdfs_root)
    print(f"Dataset split into {num_chunks} chunks")