import subprocess
import os
import json
import platform

def get_hadoop_command():
    """Get appropriate Hadoop command for OS"""
    if platform.system() == "Windows":
        hadoop_home = os.environ.get('HADOOP_HOME', 'C:/hadoop-3.3.0')
        return os.path.join(hadoop_home, 'bin', 'hdfs.cmd')
    else:
        return "hdfs"

def check_hadoop_available():
    """Kiểm tra Hadoop có sẵn không"""
    hadoop_cmd = get_hadoop_command()
    
    if not os.path.exists(hadoop_cmd):
        print(f"❌ Hadoop command not found: {hadoop_cmd}")
        print("Please check HADOOP_HOME environment variable")
        return False
    
    try:
        result = subprocess.run([hadoop_cmd, 'version'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"✅ Hadoop available: {hadoop_cmd}")
            return True
        else:
            print(f"❌ Hadoop command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing Hadoop: {e}")
        return False

def setup_hadoop_dirs():
    """Tạo các thư mục cần thiết trên HDFS"""
    if not check_hadoop_available():
        return False
        
    hadoop_cmd = get_hadoop_command()
    
    dirs = [
        '/user/tamt',
        '/user/tamt/metadata',
        '/user/tamt/chunks', 
        '/user/tamt/output',
        '/user/tamt/models'
    ]
    
    for dir_path in dirs:
        cmd = [hadoop_cmd, 'dfs', '-mkdir', '-p', dir_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print(f"✅ Created: {dir_path}")
            else:
                print(f"⚠️ {dir_path} already exists or error")
        except Exception as e:
            print(f"❌ Error creating {dir_path}: {e}")
            return False
    
    return True

def upload_and_split_dataset(dataset_name='base', num_chunks=4):
    """Upload dataset và chia thành chunks (paths đã đúng rồi)"""
    hadoop_cmd = get_hadoop_command()
    
    # Path tới file JSON local
    local_path = rf'G:\TLU\BigData\Data_time\TAMT\filelist\hmdb51-molo\{dataset_name}.json'
    
    if not os.path.exists(local_path):
        print(f"❌ File not found: {local_path}")
        return
    
    # Upload file gốc lên HDFS
    hdfs_metadata_path = f'/user/tamt/metadata/hmdb51_{dataset_name}.json'
    cmd = [hadoop_cmd, 'dfs', '-put', '-f', local_path, hdfs_metadata_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"✅ Uploaded {dataset_name}.json to HDFS")
        else:
            print(f"❌ Failed to upload {dataset_name}.json: {result.stderr}")
            return
    except Exception as e:
        print(f"❌ Error uploading {dataset_name}.json: {e}")
        return
    
    # Load data và chia chunks
    with open(local_path, 'r') as f:
        data = json.load(f)
    
    videos = data['image_names']
    labels = data.get('image_labels', [])
    chunk_size = len(videos) // num_chunks
    
    print(f"Dataset {dataset_name}: {len(videos)} videos, splitting into {num_chunks} chunks")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(videos)
        
        chunk_data = {
            'image_names': videos[start_idx:end_idx],
            'image_labels': labels[start_idx:end_idx] if labels else []
        }
        
        # Lưu chunk local
        chunk_file = f'temp_hmdb51_chunk_{i}.json'
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f)
        
        # Upload chunk lên HDFS
        hdfs_chunk_path = f'/user/tamt/chunks/hmdb51_chunk_{i}.json'
        cmd = [hadoop_cmd, 'dfs', '-put', '-f', chunk_file, hdfs_chunk_path]
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print(f"✅ Created chunk {i}: {len(chunk_data['image_names'])} videos")
        else:
            print(f"❌ Failed to upload chunk {i}: {result.stderr}")
        
        # Cleanup temp file
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

def verify_chunks():
    """Verify chunks đã upload đúng chưa"""
    hadoop_cmd = get_hadoop_command()
    
    print("\n=== Verifying Chunks ===")
    for i in range(4):
        hdfs_chunk_path = f'/user/tamt/chunks/hmdb51_chunk_{i}.json'
        cmd = [hadoop_cmd, 'dfs', '-test', '-e', hdfs_chunk_path]
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            # Get file size
            cmd_size = [hadoop_cmd, 'dfs', '-ls', hdfs_chunk_path]
            size_result = subprocess.run(cmd_size, capture_output=True, text=True, shell=True)
            print(f"✅ Chunk {i}: exists")
        else:
            print(f"❌ Chunk {i}: missing")

if __name__ == '__main__':
    print("=== HDFS Setup for HMDB51 (Simplified) ===")
    
    # Check environment
    hadoop_home = os.environ.get('HADOOP_HOME')
    java_home = os.environ.get('JAVA_HOME')
    print(f"HADOOP_HOME: {hadoop_home}")
    print(f"JAVA_HOME: {java_home}")
    
    if not hadoop_home:
        print("❌ Please set HADOOP_HOME environment variable")
        print("Example: set HADOOP_HOME=C:\\hadoop-3.3.0")
        exit(1)
    
    # Setup HDFS directories
    if setup_hadoop_dirs():
        # Upload và split dataset (paths đã đúng rồi)
        upload_and_split_dataset('base', 4)
        upload_and_split_dataset('val', 4)  # Optional: cho testing
        
        # Verify
        verify_chunks()
        
        print("\n=== HDFS Contents ===")
        hadoop_cmd = get_hadoop_command()
        try:
            subprocess.run([hadoop_cmd, 'dfs', '-ls', '-R', '/user/tamt/'], shell=True)
        except Exception as e:
            print(f"Error listing HDFS contents: {e}")
        
        print("\n✅ HDFS setup completed! Ready for MapReduce.")
    else:
        print("❌ HDFS setup failed!")