import subprocess
import os

def setup_hdfs_client():
    """Setup HDFS client để connect tới server"""
    # Cấu hình core-site.xml để connect tới HDFS server
    config = """
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://YOUR_HDFS_SERVER_IP:9000</value>
    </property>
</configuration>
"""
    
    with open('/opt/hadoop-3.3.6/etc/hadoop/core-site.xml', 'w') as f:
        f.write(config)
    
    print("✅ HDFS client configured")

def download_chunks_from_hdfs():
    """Download chunks từ HDFS server về Kaggle local"""
    os.makedirs('/kaggle/working/chunks', exist_ok=True)
    
    for i in range(4):
        hdfs_path = f'/user/tamt/chunks/hmdb51_chunk_{i}.json'
        local_path = f'/kaggle/working/chunks/chunk_{i}.json'
        
        cmd = ['hdfs', 'dfs', '-get', hdfs_path, local_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Downloaded chunk {i}")
        else:
            print(f"❌ Failed to download chunk {i}: {result.stderr}")

if __name__ == '__main__':
    setup_hdfs_client()
    download_chunks_from_hdfs()