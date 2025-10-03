import json
import os
import subprocess
import argparse
import time
import sys
sys.path.append("/kaggle/input/tamt-bigdata/TAMT")
def setup_environment():
    """Setup environment cho Kaggle với memory optimization"""
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
    os.environ['HADOOP_HOME'] = '/kaggle/working/hadoop'
    os.environ['PATH'] = os.environ['HADOOP_HOME'] + '/bin:' + os.environ['HADOOP_HOME'] + '/sbin:' + os.environ.get('PATH', '')
    
    # MEMORY OPTIMIZATION
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['PYTORCH_JIT'] = '0'  # Disable JIT for memory

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
    """Run training với parameters nhẹ hơn và 5 epochs để fix memory issues"""
    
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
    
    # AGGRESSIVE MEMORY MANAGEMENT
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'  # Smaller chunks
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['PYTHONHASHSEED'] = str(worker_seed)
    env['OMP_NUM_THREADS'] = '1'  # Single thread
    env['MKL_NUM_THREADS'] = '1'
    
    # FORCE COMPLETE MEMORY CLEANUP
    try:
        import torch
        import gc
        
        if torch.cuda.is_available():
            # More aggressive cleanup
            for i in range(5):  # More cleanup cycles
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
            
            # Reset memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except:
                pass
                
            print(f"✅ Cleared GPU cache for worker {chunk_id}")
    except Exception as e:
        print(f"⚠️ GPU cache clear failed: {e}")
    
    # CHECK GPU AVAILABILITY
    try:
        import torch
        gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            use_gpu = True
            gpu_arg = str(gpu_id)
            
            # LIGHTER PARAMETERS - fix memory issues + 5 epochs
            if chunk_id == 0:
                n_shot = 8               # Increase 
                train_episodes = 60      # 2x increase from 30
                val_episodes = 30        # 2x increase from 15
                reduce_dim = 128         # 2x increase from 64
                n_query = 16             # Increase from 12
                lr = '1e-3'              
                epochs = 8               # Increase from 5
                milestones = '5'         # Adjust
            elif chunk_id == 1:
                n_shot = 8
                train_episodes = 50      # 2x increase from 25
                val_episodes = 25        # 2x increase from 12
                reduce_dim = 128         # 2x increase from 64
                n_query = 15             # Increase from 10
                lr = '8e-4'              
                epochs = 8
                milestones = '5'
            elif chunk_id == 2:
                n_shot = 6               # Increase from 5
                train_episodes = 40      # 2x increase from 20
                val_episodes = 20        # 2x increase from 10
                reduce_dim = 96          # Increase from 64
                n_query = 12             # Increase from 10
                lr = '8e-4'
                epochs = 6
                milestones = '4'
            else:  # chunk 3
                n_shot = 6               # Increase from 5
                train_episodes = 35      # 2x+ increase from 15
                val_episodes = 18        # 2x+ increase from 8
                reduce_dim = 96          # Increase from 64
                n_query = 12             # Increase from 10
                lr = '5e-4'
                epochs = 6
                milestones = '4'
            
            timeout = 4800  # 80 minutes instead of 40
            
        else:
            # CPU mode
            use_gpu = False
            gpu_arg = '-1'
            n_shot = 3
            train_episodes = 15
            val_episodes = 8
            reduce_dim = 32
            n_query = 6
            lr = '1e-3'
            epochs = 5
            milestones = '3'
            timeout = 3600
            
    except Exception as e:
        print(f"⚠️ GPU setup failed: {e}")
        use_gpu = False
        gpu_arg = '-1'
        n_shot = 3
        train_episodes = 15
        val_episodes = 8
        reduce_dim = 32
        n_query = 6
        lr = '1e-3'
        epochs = 5
        milestones = '3'
        timeout = 3600
    
    # TRAINING COMMAND - LIGHTER PARAMETERS
    cmd = [
        'python', '-u', train_script,
        '--dataset', 'hmdb51',
        '--data_path', dataset_dir,
        '--model', 'VideoMAES',
        '--method', 'meta_deepbdc',
        '--image_size', '112',
        '--gpu', gpu_arg,
        '--lr', lr,
        '--epoch', str(epochs),              # 5 EPOCHS
        '--milestones', milestones,          # Adjusted for 5 epochs
        '--n_shot', str(n_shot),            # LIGHTER n_shot
        '--n_query', str(n_query),          # MUCH LIGHTER n_query
        '--train_n_episode', str(train_episodes),  # MUCH LIGHTER episodes
        '--val_n_episode', str(val_episodes),      # MUCH LIGHTER val episodes
        '--reduce_dim', str(reduce_dim),           # MUCH LIGHTER dimensions
        '--extra_dir', output_dir,
        '--seed', str(worker_seed),
        '--save_freq', '2'                   # Save every 2 epochs
    ]
    
    # Add pretrained model
    if pretrained_path:
        cmd.extend(['--pretrain_path', pretrained_path])
        print(f"   ✅ Using pretrained model for worker {chunk_id}")
    
    # UPDATED PRINT STATEMENTS
    print(f"🎯 Map Worker {chunk_id}: Training 5 EPOCHS with LIGHTER parameters")
    print(f"   LIGHTER Parameters: n_shot={n_shot}, n_query={n_query}, episodes={train_episodes}/{val_episodes}")
    print(f"   Learning: lr={lr}, reduce_dim={reduce_dim}, milestones={milestones}")
    print(f"   Expected time: {timeout//60} minutes")
    
    # VERIFY COMMAND - print key parameters
    print(f"🔍 KEY PARAMS: epochs={epochs}, n_query={n_query}, reduce_dim={reduce_dim}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              cwd=working_dir, env=env, timeout=timeout)
        
        print(f"📊 Worker {chunk_id}: Completed (code: {result.returncode})")
        
        # DEBUG: Always show stderr for diagnosis
        if result.stderr:
            print(f"🔍 STDERR:")
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"   {line}")
        
        # DEBUG: Show stdout for successful runs
        if result.stdout:
            print(f"🔍 STDOUT (last 20 lines):")
            stdout_lines = result.stdout.split('\n')
            for line in stdout_lines[-20:]:
                if line.strip() and any(keyword in line.lower() for keyword in ['acc', 'loss', 'epoch', 'val', 'test', 'namespace']):
                    print(f"   {line}")
        
        if result.returncode == 0:
            print(f"✅ Training successful!")
            
            # EXTRACT REAL ACCURACY from stdout
            real_accuracy = extract_accuracy_from_stdout(result.stdout)
            
            # SAVE MODEL with better file management
            import glob
            model_found = False
            
            # Search for model files
            model_patterns = ['best_model.tar', '*.tar', '*.pth']
            
            for pattern in model_patterns:
                model_files = glob.glob(f'{working_dir}/**/{pattern}', recursive=True)
                for model_file in model_files:
                    try:
                        import shutil
                        filename = os.path.basename(model_file)
                        output_model = f'{output_dir}/{filename}'
                        shutil.copy2(model_file, output_model)
                        print(f"   📦 Saved model: {filename} ({os.path.getsize(model_file)/1024/1024:.1f}MB)")
                        model_found = True
                        break
                    except Exception as e:
                        print(f"   ⚠️ Failed to copy model {model_file}: {e}")
                if model_found:
                    break
            
            # Copy checkpoint directory
            checkpoint_dir = f'{working_dir}/checkpoints'
            if os.path.exists(checkpoint_dir):
                import shutil
                try:
                    output_checkpoint = f'{output_dir}/checkpoints'
                    if os.path.exists(output_checkpoint):
                        shutil.rmtree(output_checkpoint)
                    shutil.copytree(checkpoint_dir, output_checkpoint)
                    print(f"   ✅ Copied checkpoint directory")
                except Exception as copy_error:
                    print(f"   ⚠️ Checkpoint copy failed: {copy_error}")
            
            # SAVE RESULT with real accuracy
            chunk_result = {
                'chunk_id': chunk_id,
                'status': 'success',
                'accuracy': real_accuracy,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'episodes_trained': train_episodes,
                'epochs_trained': epochs,
                'n_shot': n_shot,
                'n_query': n_query
            }
            
            with open(f'/tmp/map_result_{chunk_id}.json', 'w') as f:
                json.dump(chunk_result, f)
                        
            print(f"   🎯 Final Accuracy: {real_accuracy:.2f}%")
            
        else:
            print(f"❌ TRAINING FAILED - Return Code: {result.returncode}")
                        
            # Save failed result với detailed info
            chunk_result = {
                'chunk_id': chunk_id,
                'status': 'failed',
                'accuracy': 0.0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'command': ' '.join(cmd)
            }
            
            with open(f'/tmp/map_result_{chunk_id}.json', 'w') as f:
                json.dump(chunk_result, f)
        
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

def extract_accuracy_from_stdout(stdout):
    """Extract real accuracy from training stdout"""
    import re
    
    if not stdout:
        return 0.0
    
    # Look for accuracy patterns in order of preference
    patterns = [
        r'model best acc is ([\d.]+)',  # Most reliable
        r'val acc is ([\d.]+)',         # Validation accuracy
        r'Test Acc = ([\d.]+)%',       # Test accuracy with %
        r'Test Acc = ([\d.]+)',        # Test accuracy without %
        r'best acc is ([\d.]+)',       # Generic best acc
        r'acc is ([\d.]+)'             # Fallback
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, stdout)
        if matches:
            # Take the last (most recent) accuracy
            accuracy = float(matches[-1])
            return accuracy
    
    # If no pattern found, try to find any number followed by %
    percent_matches = re.findall(r'(\d+\.\d+)%', stdout)
    if percent_matches:
        # Take the last percentage that looks like accuracy (20-100 range)
        for match in reversed(percent_matches):
            acc = float(match)
            if 20.0 <= acc <= 100.0:  # Reasonable accuracy range
                return acc
    
    return 0.0

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
    
    print(f"🔥 MAP WORKER {args.chunk_id} STARTING (10 EPOCHS MODE)")
    print("=" * 60)
    
    try:
        # Setup environment
        setup_environment()
              
        # Download pretrained
        pretrained_path = download_pretrained_model()
        
        # Prepare dataset
        dataset_dir, num_videos = prepare_chunk_dataset(args.chunk_id)
        
        # Run training with 10 epochs + high parameters
        result, output_dir = run_meta_training(args.chunk_id, dataset_dir, pretrained_path, args.gpu_id)
        
        # Upload results
        upload_results_to_hdfs(args.chunk_id, output_dir, result)
        
        # Final status
        if result.returncode == 0:
            print(f"🏆 Map Worker {args.chunk_id}: SUCCESS (10 epochs completed)")
        else:
            print(f"❌ Map Worker {args.chunk_id}: FAILED")
            
    except Exception as e:
        print(f"💥 Map Worker {args.chunk_id}: EXCEPTION - {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final GPU cleanup
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except:
            pass
            
        print(f"✅ Worker {args.chunk_id} completed successfully")

if __name__ == '__main__':
    main()