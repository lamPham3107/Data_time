import json
import os
import pickle
import subprocess
import sys
import argparse

def create_temp_dataset(chunk_id, hdfs_root):
    """Tạo dataset tạm cho chunk"""
    chunk_path = f"{hdfs_root}/input/chunk_{chunk_id}.json"
    
    with open(chunk_path, 'r') as f:
        chunk_data = json.load(f)
    
    temp_data_dir = f"{hdfs_root}/temp_chunk_{chunk_id}"
    os.makedirs(temp_data_dir, exist_ok=True)
    
    with open(f"{temp_data_dir}/base.json", 'w') as f:
        json.dump(chunk_data, f)
    
    return temp_data_dir, len(chunk_data['image_names'])

def run_training(chunk_id, temp_data_dir, model_path, hdfs_root, gpu_id=0):
    """Chạy training cho chunk"""
    output_dir = f"{hdfs_root}/map_output/chunk_{chunk_id}_model"
    
    cmd = [
        'python', 'meta_train.py',
        '--dataset', 'ucf101',
        '--data_path', temp_data_dir,
        '--model', 'VideoMAES',
        '--method', 'meta_deepbdc',
        '--image_size', '112',
        '--gpu', str(gpu_id),
        '--lr', '1e-3',
        '--epoch', '10',
        '--milestones', '5',
        '--n_shot', '5',
        '--train_n_episode', '100',
        '--val_n_episode', '50',
        '--reduce_dim', '256',
        '--pretrain_path', model_path,
        '--extra_dir', output_dir
    ]
    
    print(f"Map Worker {chunk_id}: Starting training...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result

def save_map_result(chunk_id, result, hdfs_root, num_videos):
    """Lưu kết quả map worker"""
    map_result = {
        'chunk_id': chunk_id,
        'num_videos': num_videos,
        'status': 'completed' if result.returncode == 0 else 'failed',
        'stdout': result.stdout[-1000:],  # Lưu 1000 ký tự cuối
        'stderr': result.stderr[-1000:] if result.stderr else '',
        'return_code': result.returncode
    }
    
    with open(f"{hdfs_root}/map_output/map_result_{chunk_id}.pkl", 'wb') as f:
        pickle.dump(map_result, f)
    
    print(f"Map Worker {chunk_id}: {'Completed' if result.returncode == 0 else 'Failed'}")
    return map_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_id', type=int, required=True)
    parser.add_argument('--hdfs_root', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    # Tạo temp dataset
    temp_data_dir, num_videos = create_temp_dataset(args.chunk_id, args.hdfs_root)
    print(f"Map Worker {args.chunk_id}: Processing {num_videos} videos")
    
    # Chạy training
    result = run_training(args.chunk_id, temp_data_dir, args.model_path, args.hdfs_root, args.gpu_id)
    
    # Lưu kết quả
    save_map_result(args.chunk_id, result, args.hdfs_root, num_videos)

if __name__ == '__main__':
    main()