# Fix scripts để dùng đúng Hadoop path trong Kaggle
import os

print("🔧 Fixing script paths for Kaggle environment...")

# Fix map_worker.py
map_worker_path = '/kaggle/working/map_worker.py'
if os.path.exists(map_worker_path):
    with open(map_worker_path, 'r') as f:
        content = f.read()
    
    # Replace hardcoded paths
    content = content.replace('/opt/hadoop-3.3.6', '/opt/hadoop-3.3.0')
    content = content.replace('C:/hadoop-3.3.0', '/opt/hadoop-3.3.0')
    content = content.replace('C:/Java/jdk1.8.0_202', '/usr/lib/jvm/java-11-openjdk-amd64')
    
    # Fix Windows paths to Linux
    content = content.replace('\\\\', '/')
    content = content.replace('\\', '/')
    
    with open(map_worker_path, 'w') as f:
        f.write(content)
    print("✅ Fixed map_worker.py")

# Fix reduce_worker.py  
reduce_worker_path = '/kaggle/working/reduce_worker.py'
if os.path.exists(reduce_worker_path):
    with open(reduce_worker_path, 'r') as f:
        content = f.read()
    
    content = content.replace('/opt/hadoop-3.3.6', '/opt/hadoop-3.3.0')
    content = content.replace('C:/hadoop-3.3.0', '/opt/hadoop-3.3.0')
    content = content.replace('C:/Java/jdk1.8.0_202', '/usr/lib/jvm/java-11-openjdk-amd64')
    
    with open(reduce_worker_path, 'w') as f:
        f.write(content)
    print("✅ Fixed reduce_worker.py")

# Fix run_mapreduce.sh
mapreduce_script_path = '/kaggle/working/run_mapreduce.sh'
if os.path.exists(mapreduce_script_path):
    with open(mapreduce_script_path, 'r') as f:
        content = f.read()
    
    # Fix environment variables for Linux
    content = content.replace('export HADOOP_HOME=C:/hadoop-3.3.0', 'export HADOOP_HOME=/opt/hadoop-3.3.0')
    content = content.replace('export JAVA_HOME=C:/Java/jdk1.8.0_202', 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64')
    content = content.replace('hadoop-3.3.6', 'hadoop-3.3.0')
    
    with open(mapreduce_script_path, 'w') as f:
        f.write(content)
    print("✅ Fixed run_mapreduce.sh")

# Set environment variables properly
os.environ['HADOOP_HOME'] = '/opt/hadoop-3.3.0'
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
os.environ['PATH'] = f"/opt/hadoop-3.3.0/bin:{os.environ.get('PATH', '')}"

print("✅ All scripts updated for Kaggle environment!")
print(f"HADOOP_HOME: {os.environ['HADOOP_HOME']}")
print(f"JAVA_HOME: {os.environ['JAVA_HOME']}")