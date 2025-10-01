import subprocess
import os
import platform

def get_hadoop_command():
    """Get appropriate Hadoop command for OS"""
    if platform.system() == "Windows":
        hadoop_home = os.environ.get('HADOOP_HOME', 'C:/hadoop-3.3.0')
        return os.path.join(hadoop_home, 'bin', 'hdfs.cmd')
    else:
        return "hdfs"

def upload_pretrained_to_hdfs():
    """Upload pretrained model lên HDFS"""
    hadoop_cmd = get_hadoop_command()
    
    # Local pretrained model path
    local_model_path = r'G:\TLU\BigData\Data_time\TAMT\checkpoints\hmdb51\best_model.tar'
    
    # Check file exists
    if not os.path.exists(local_model_path):
        print(f"❌ Pretrained model not found: {local_model_path}")
        return False
    
    # File size
    file_size = os.path.getsize(local_model_path) / (1024*1024)  # MB
    print(f"📁 Model size: {file_size:.2f} MB")
    
    # Create HDFS directory
    hdfs_dir = '/user/tamt/models'
    cmd = [hadoop_cmd, 'dfs', '-mkdir', '-p', hdfs_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    
    # Upload model
    hdfs_model_path = '/user/tamt/models/hmdb51_pretrained.tar'
    cmd = [hadoop_cmd, 'dfs', '-put', '-f', local_model_path, hdfs_model_path]
    
    print(f"🚀 Uploading pretrained model to HDFS...")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    
    if result.returncode == 0:
        print(f"✅ Upload successful: {hdfs_model_path}")
        
        # Verify upload
        cmd_ls = [hadoop_cmd, 'dfs', '-ls', hdfs_model_path]
        ls_result = subprocess.run(cmd_ls, capture_output=True, text=True, shell=True)
        if ls_result.returncode == 0:
            print(f"✅ Verification: {ls_result.stdout.strip()}")
        
        return True
    else:
        print(f"❌ Upload failed: {result.stderr}")
        return False

if __name__ == '__main__':
    print("=== Upload Pretrained Model to HDFS ===")
    upload_pretrained_to_hdfs()