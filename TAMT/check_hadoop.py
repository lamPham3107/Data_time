# check_hadoop.py - với đúng paths
import subprocess
import os

def check_hadoop_setup():
    """Kiểm tra Hadoop setup"""
    
    # Expected paths
    expected_hadoop = 'C:/hadoop-3.3.0'  # Fix
    expected_java = 'C:/Java/jdk1.8.0_202'
    
    hadoop_home = os.environ.get('HADOOP_HOME')
    java_home = os.environ.get('JAVA_HOME')
    
    print(f"Expected HADOOP_HOME: {expected_hadoop}")
    print(f"Actual HADOOP_HOME: {hadoop_home}")
    print(f"Expected JAVA_HOME: {expected_java}")
    print(f"Actual JAVA_HOME: {java_home}")
    
    # Check Hadoop binary với đúng path
    hadoop_bin = os.path.join(expected_hadoop, 'bin', 'hdfs.cmd')
    if os.path.exists(hadoop_bin):
        print(f"✅ Hadoop binary found: {hadoop_bin}")
    else:
        print(f"❌ Hadoop binary not found: {hadoop_bin}")
        return False
    
    # Test command
    try:
        result = subprocess.run([hadoop_bin, 'version'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✅ Hadoop command works")
            print(result.stdout.split('\n')[0])
        else:
            print(f"❌ Hadoop command failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    check_hadoop_setup()