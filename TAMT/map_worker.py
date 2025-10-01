import json
import os
import subprocess
import argparse
import time
import sys
sys.path.append("/kaggle/input/tamt-bigdata/TAMT")
def setup_environment():
    """Setup environment cho Kaggle"""
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
    os.environ['HADOOP_HOME'] = '/kaggle/working/hadoop'
    os.environ['PATH'] = os.environ['HADOOP_HOME'] + '/bin:' + os.environ['HADOOP_HOME'] + '/sbin:' + os.environ.get('PATH', '')

def setup_hdfs_client_connection():
    """Setup HDFS client"""
    config = """<?xml version="1.0"?>
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>"""
    config_path = '/kaggle/working/hadoop/etc/hadoop/core-site.xml'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config)

def run_hdfs_command(cmd_args, retries=3, retry_delay=0.5):
    """Run HDFS command với retry logic"""
    hadoop_home = os.environ.get('HADOOP_HOME', '/kaggle/working/hadoop')
    hdfs_cmd = [f'{hadoop_home}/bin/hdfs'] + cmd_args
    
    for attempt in range(retries):
        try:
            result = subprocess.run(hdfs_cmd, capture_output=True, text=True, timeout=30)
            return result
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                time.sleep(retry_delay)
                continue
            return subprocess.CompletedProcess(args=hdfs_cmd, returncode=1, stdout='', stderr='HDFS timeout')
        except Exception as e:
            return subprocess.CompletedProcess(args=hdfs_cmd, returncode=1, stdout='', stderr=str(e))

def download_chunk_metadata(chunk_id):
    """Download chunk metadata từ HDFS với fallback"""
    hdfs_chunk_path = f'/user/tamt/chunks/hmdb51_chunk_{chunk_id}.json'
    final_chunk_path = f'/tmp/chunk_{chunk_id}.json'
    
    # Try HDFS first
    result = run_hdfs_command(['dfs', '-get', hdfs_chunk_path, final_chunk_path])
    
    if result.returncode == 0 and os.path.exists(final_chunk_path):
        print(f"✅ Downloaded chunk {chunk_id} from HDFS")
        return final_chunk_path
    
    # Fallback: Create chunk from local video files
    print(f"⚠️ HDFS failed, creating chunk {chunk_id} from local videos...")
    return create_chunk_from_local_videos(chunk_id)

def create_chunk_from_local_videos(chunk_id):
    """Tạo chunk từ video files local"""
    hmdb51_path = '/kaggle/input/data-bigdata/data_down/hmdb51_org_2'
    
    if not os.path.exists(hmdb51_path):
        raise Exception(f"HMDB51 videos not found at {hmdb51_path}")
    
    # Get all video files
    all_videos = []
    for root, dirs, files in os.walk(hmdb51_path):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mov')):
                video_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                all_videos.append({
                    'path': video_path,
                    'class': class_name
                })
    
    print(f"📊 Found {len(all_videos)} total videos")
    
    # Split into chunks
    chunk_size = len(all_videos) // 4  # 4 chunks
    start_idx = chunk_id * chunk_size
    end_idx = start_idx + chunk_size if chunk_id < 3 else len(all_videos)
    
    chunk_videos = all_videos[start_idx:end_idx]
    
    # Create chunk data structure
    chunk_data = {
        "image_names": [v['path'] for v in chunk_videos],
        "image_labels": [hash(v['class']) % 51 for v in chunk_videos],  # Simple class mapping
        "total_videos": len(chunk_videos)
    }
    
    # Save chunk
    chunk_file = f'/tmp/chunk_{chunk_id}.json'
    with open(chunk_file, 'w') as f:
        json.dump(chunk_data, f)
    
    print(f"✅ Created chunk {chunk_id} with {len(chunk_videos)} videos")
    return chunk_file

def prepare_chunk_dataset(chunk_id):
    """Chuẩn bị dataset cho training"""
    # Download/create chunk metadata
    chunk_file = download_chunk_metadata(chunk_id)
    
    with open(chunk_file, 'r') as f:
        chunk_data = json.load(f)
    
    # Create dataset directory
    temp_dir = f'/tmp/chunk_{chunk_id}_dataset'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create base.json và val.json cho meta training
    videos = chunk_data['image_names']
    labels = chunk_data.get('image_labels', list(range(len(videos))))
    
    # Split 80% train, 20% val
    split_idx = int(len(videos) * 0.8)
    
    train_data = {
        "image_names": videos[:split_idx],
        "image_labels": labels[:split_idx]
    }
    
    val_data = {
        "image_names": videos[split_idx:],
        "image_labels": labels[split_idx:]
    }
    
    # Save datasets
    with open(f'{temp_dir}/base.json', 'w') as f:
        json.dump(train_data, f)
    
    with open(f'{temp_dir}/val.json', 'w') as f:
        json.dump(val_data, f)
    
    print(f"📊 Chunk {chunk_id}: {len(train_data['image_names'])} train, {len(val_data['image_names'])} val videos")
    
    return temp_dir, len(videos)

def download_pretrained_model():
    """Download pretrained model"""
    # Try HDFS first
    hdfs_model_path = '/user/tamt/models/hmdb51_pretrained.tar'
    local_model_path = f'/tmp/pretrained_model_{os.getpid()}.tar'
    
    result = run_hdfs_command(['dfs', '-get', hdfs_model_path, local_model_path])
    
    if result.returncode == 0 and os.path.exists(local_model_path):
        print(f"✅ Downloaded pretrained from HDFS: {os.path.getsize(local_model_path)/1024/1024:.1f}MB")
        return local_model_path
    
    # Try Kaggle paths
    kaggle_paths = [
        '/kaggle/input/tamt-bigdata/TAMT/checkpoints/hmdb51/best_model.tar',
        '/kaggle/input/tamt-bigdata/TAMT/checkpoints/best_model.tar',
        '/kaggle/input/hmdb51-pretrained/best_model.tar'
    ]
    
    for path in kaggle_paths:
        if os.path.exists(path):
            print(f"✅ Found pretrained in Kaggle: {path}")
            return path
    
    print("⚠️ No pretrained model found - training from scratch")
    return None

def run_meta_training(chunk_id, dataset_dir, pretrained_path, gpu_id=0):
    """Run training with proper GPU detection"""
    
    # Set unique environment
    worker_seed = 1000 + chunk_id
    os.environ['WORKER_ID'] = str(chunk_id)
    
    output_dir = f'/tmp/map_output_chunk_{chunk_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create writable working directory
    working_dir = f'/tmp/working_chunk_{chunk_id}'
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(f'{working_dir}/checkpoints/hmdb51', exist_ok=True)
    
    tamt_dir = '/kaggle/input/tamt-bigdata/TAMT'
    train_script = f'{tamt_dir}/meta_train.py'
    
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{tamt_dir}:' + env.get('PYTHONPATH', '')
    
    # CHECK GPU AVAILABILITY
    try:
        import torch
        gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            
            # ENABLE GPU
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            use_gpu = True
            gpu_arg = str(gpu_id)
            
            # GPU training parameters
            n_shot = 5
            train_episodes = 10
            val_episodes = 5
            reduce_dim = 64
            timeout = 600  # 10 minutes
            
        else:
            print(f"⚠️ No GPU available, using CPU")
            env['CUDA_VISIBLE_DEVICES'] = ''
            use_gpu = False
            gpu_arg = '-1'
            
            # CPU training parameters (reduced)
            n_shot = 3
            train_episodes = 3
            val_episodes = 2
            reduce_dim = 32
            timeout = 1800  # 30 minutes
            
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}, using CPU")
        env['CUDA_VISIBLE_DEVICES'] = ''
        use_gpu = False
        gpu_arg = '-1'
        n_shot = 3
        train_episodes = 3
        val_episodes = 2
        reduce_dim = 32
        timeout = 1800
    
    # TRAINING COMMAND with dynamic GPU/CPU parameters
    cmd = [
        'python', '-u', train_script,
        '--dataset', 'hmdb51',
        '--data_path', dataset_dir,
        '--model', 'VideoMAES',
        '--method', 'meta_deepbdc',
        '--image_size', '112',
        '--gpu', gpu_arg,                   # Dynamic GPU/CPU
        '--lr', '1e-3',
        '--epoch', '1',                     # Minimal for testing
        '--n_shot', str(n_shot),            # Dynamic based on device
        '--train_n_episode', str(train_episodes),  # Dynamic
        '--val_n_episode', str(val_episodes),      # Dynamic
        '--reduce_dim', str(reduce_dim),           # Dynamic
        '--extra_dir', output_dir,
        '--seed', str(worker_seed),
        '--save_freq', '1'
    ]
    
    if pretrained_path:
        cmd.extend(['--pretrain_path', pretrained_path])
    
    print(f"🚀 Map Worker {chunk_id}: Starting {'GPU' if use_gpu else 'CPU'} training...")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Working dir: {working_dir}")
    print(f"   Device: {'GPU' if use_gpu else 'CPU'} (arg: {gpu_arg})")
    print(f"   Episodes: {train_episodes} train, {val_episodes} val")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              cwd=working_dir, env=env, timeout=timeout)
        
        print(f"📊 Worker {chunk_id}: Completed (code: {result.returncode})")
        
        if result.returncode != 0:
            print(f"❌ TRAINING FAILED - Error Details:")
            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                # Show relevant error lines
                for line in stderr_lines[-8:]:
                    if line.strip() and any(keyword in line for keyword in ['Error', 'Exception', 'ValueError', 'RuntimeError']):
                        print(f"   {line}")
            
            if result.stdout:
                stdout_lines = result.stdout.strip().split('\n')
                print(f"   Last stdout:")
                for line in stdout_lines[-3:]:
                    if line.strip():
                        print(f"   {line}")
        else:
            print(f"✅ Training successful!")
            
            # Show training metrics
            if result.stdout:
                stdout_lines = result.stdout.split('\n')
                for line in stdout_lines[-10:]:
                    if any(keyword in line.lower() for keyword in ['acc', 'loss', 'epoch', 'val']):
                        print(f"   {line}")
        
        return result, output_dir
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Worker {chunk_id}: Training timeout ({timeout//60} min)")
        return subprocess.CompletedProcess(
            args=cmd, returncode=124, stdout='', stderr='Training timeout'
        ), output_dir
    except Exception as e:
        print(f"💥 Worker {chunk_id}: Exception - {e}")
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout='', stderr=str(e)
        ), output_dir

def upload_results_to_hdfs(chunk_id, output_dir, result):
    """Upload kết quả lên HDFS"""
    # Create result metadata
    map_result = {
        'chunk_id': chunk_id,
        'status': 'success' if result.returncode == 0 else 'failed',
        'return_code': result.returncode,
        'stdout': result.stdout[-1000:] if result.stdout else '',
        'stderr': result.stderr[-500:] if result.stderr else ''
    }
    
    # Save result locally first
    metadata_file = f'/tmp/map_result_{chunk_id}.json'
    with open(metadata_file, 'w') as f:
        json.dump(map_result, f)
    
    # Try upload to HDFS
    hdfs_metadata_path = f'/user/tamt/output/map_result_{chunk_id}.json'
    run_hdfs_command(['dfs', '-rm', '-f', hdfs_metadata_path])
    upload_result = run_hdfs_command(['dfs', '-put', metadata_file, hdfs_metadata_path])
    
    if upload_result.returncode == 0:
        print(f"✅ Uploaded result metadata for chunk {chunk_id}")
    
    # Upload model if successful
    if result.returncode == 0:
        model_file = f'{output_dir}/best_model.tar'
        if os.path.exists(model_file):
            hdfs_model_path = f'/user/tamt/output/chunk_{chunk_id}_best_model.tar'
            run_hdfs_command(['dfs', '-rm', '-f', hdfs_model_path])
            model_upload = run_hdfs_command(['dfs', '-put', model_file, hdfs_model_path])
            
            if model_upload.returncode == 0:
                size = os.path.getsize(model_file) / (1024*1024)
                print(f"✅ Uploaded model for chunk {chunk_id} ({size:.1f}MB)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_id', type=int, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    print(f"🔥 MAP WORKER {args.chunk_id} STARTING")
    print("=" * 50)
    
    try:
        # Setup environment
        setup_environment()
              
        # Download pretrained
        pretrained_path = download_pretrained_model()
        
        # Prepare dataset
        dataset_dir, num_videos = prepare_chunk_dataset(args.chunk_id)
        
        # Run training
        result, output_dir = run_meta_training(args.chunk_id, dataset_dir, pretrained_path, args.gpu_id)
        
        # Upload results
        upload_results_to_hdfs(args.chunk_id, output_dir, result)
        
        # Final status
        if result.returncode == 0:
            print(f"🏆 Map Worker {args.chunk_id}: SUCCESS")
        else:
            print(f"❌ Map Worker {args.chunk_id}: FAILED")
            
    except Exception as e:
        print(f"💥 Map Worker {args.chunk_id}: EXCEPTION - {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()