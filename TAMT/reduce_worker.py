import json
import os
import subprocess
import argparse
import torch
import glob
import shutil
import time
import re

def setup_environment():
    """Setup environment"""
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
    os.environ['HADOOP_HOME'] = '/kaggle/working/hadoop'
    os.environ['PATH'] = os.environ['HADOOP_HOME'] + '/bin:' + os.environ['HADOOP_HOME'] + '/sbin:' + os.environ.get('PATH', '')

def run_hdfs_command(cmd_args, retries=3):
    """Run HDFS command"""
    hadoop_home = os.environ.get('HADOOP_HOME', '/kaggle/working/hadoop')
    hdfs_cmd = [f'{hadoop_home}/bin/hdfs'] + cmd_args
    
    for attempt in range(retries):
        try:
            result = subprocess.run(hdfs_cmd, capture_output=True, text=True, timeout=30)
            return result
        except:
            if attempt < retries - 1:
                continue
            return subprocess.CompletedProcess(args=hdfs_cmd, returncode=1, stdout='', stderr='HDFS failed')

def collect_map_results(num_chunks):
    """Thu thập kết quả từ map workers - REAL RESULTS ONLY"""
    map_results = []
    successful_chunks = []
    
    print("🔍 Collecting Map results...")
    
    for i in range(num_chunks):
        # Check local result files from map workers
        local_result_path = f'/tmp/map_result_{i}.json'
        
        if os.path.exists(local_result_path):
            try:
                with open(local_result_path, 'r') as f:
                    map_result = json.load(f)
                    map_results.append(map_result)
                    
                    if map_result.get('status') == 'success':
                        successful_chunks.append(i)
                        print(f"✅ Chunk {i}: SUCCESS - Real Acc: {map_result.get('accuracy', 'N/A')}%")
                    else:
                        print(f"⚠️ Chunk {i}: FAILED - {map_result.get('stderr', '')[:100]}...")
                        
            except Exception as e:
                print(f"❌ Error reading result for chunk {i}: {e}")
        else:
            print(f"❌ No result found for chunk {i}")
    
    return map_results, successful_chunks

def extract_real_accuracy_from_training_log(chunk_id):
    """Extract REAL accuracy from training stdout/logs"""
    
    # Try to find training logs in multiple locations
    log_locations = [
        f'/tmp/map_result_{chunk_id}.json',
        f'/tmp/map_output_chunk_{chunk_id}/training.log',
        f'/tmp/working_chunk_{chunk_id}/training.log'
    ]
    
    real_accuracy = None
    
    # Method 1: From map result JSON
    result_file = f'/tmp/map_result_{chunk_id}.json'
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                
            # Extract from stdout if available
            stdout = result_data.get('stdout', '')
            if stdout:
                # Look for patterns like "val acc is 45.00" or "Test Acc = 45.00%"
                import re
                
                # Pattern 1: "val acc is XX.XX"
                match = re.search(r'val acc is ([\d.]+)', stdout)
                if match:
                    real_accuracy = float(match.group(1))
                    print(f"   📊 Extracted val acc: {real_accuracy:.2f}%")
                    return real_accuracy
                
                # Pattern 2: "Test Acc = XX.XX%"
                match = re.search(r'Test Acc = ([\d.]+)%', stdout)
                if match:
                    real_accuracy = float(match.group(1))
                    print(f"   📊 Extracted test acc: {real_accuracy:.2f}%")
                    return real_accuracy
                
                # Pattern 3: "model best acc is XX.XX"
                match = re.search(r'model best acc is ([\d.]+)', stdout)
                if match:
                    real_accuracy = float(match.group(1))
                    print(f"   📊 Extracted best acc: {real_accuracy:.2f}%")
                    return real_accuracy
                    
        except Exception as e:
            print(f"   ⚠️ Error parsing result file: {e}")
    
    return None

def evaluate_model_performance(model_info, map_results):
    """Đánh giá performance từ REAL training results"""
    chunk_id = model_info['chunk_id']
    
    print(f"📊 Evaluating Chunk {chunk_id} with REAL results...")
    
    # Method 1: Extract from training logs
    real_accuracy = extract_real_accuracy_from_training_log(chunk_id)
    
    if real_accuracy is not None:
        print(f"   ✅ Real accuracy: {real_accuracy:.2f}%")
        return {
            'chunk_id': chunk_id,
            'accuracy': float(real_accuracy),
            'model_path': model_info['model_path'],
            'model_size': model_info['size'],
            'source': 'training_log'
        }
    
    # Method 2: Find corresponding map result
    for map_result in map_results:
        if map_result.get('chunk_id') == chunk_id:
            if 'accuracy' in map_result:
                accuracy = float(map_result['accuracy'])
                print(f"   ✅ Accuracy from map result: {accuracy:.2f}%")
                return {
                    'chunk_id': chunk_id,
                    'accuracy': accuracy,
                    'model_path': model_info['model_path'],
                    'model_size': model_info['size'],
                    'source': 'map_result'
                }
    
    # Method 3: Try to load model checkpoint
    try:
        checkpoint = torch.load(model_info['model_path'], map_location='cpu')
        
        if isinstance(checkpoint, dict):
            # Look for accuracy keys in checkpoint
            accuracy_keys = ['accuracy', 'val_acc', 'best_acc', 'test_acc']
            for key in accuracy_keys:
                if key in checkpoint:
                    accuracy = float(checkpoint[key])
                    print(f"   ✅ Accuracy from checkpoint[{key}]: {accuracy:.2f}%")
                    return {
                        'chunk_id': chunk_id,
                        'accuracy': accuracy,
                        'model_path': model_info['model_path'],
                        'model_size': model_info['size'],
                        'source': f'checkpoint_{key}'
                    }
                    
    except Exception as e:
        print(f"   ⚠️ Could not load checkpoint: {e}")
    
    # If no real accuracy found, mark as failed
    print(f"   ❌ No real accuracy found for chunk {chunk_id}")
    return {
        'chunk_id': chunk_id,
        'accuracy': 0.0,  # Mark as failed
        'model_path': model_info['model_path'],
        'model_size': model_info['size'],
        'source': 'failed_to_extract'
    }

def select_best_model(test_results):
    """Chọn model tốt nhất"""
    if not test_results:
        return None
    
    # Sort by accuracy
    best_result = max(test_results, key=lambda x: x['accuracy'])
    
    print(f"🏆 Best model: Chunk {best_result['chunk_id']} with {best_result['accuracy']:.2f}% accuracy")
    
    return best_result

def create_test_dataset():
    """Tạo test dataset từ video files"""
    hmdb51_path = '/kaggle/input/data-bigdata/data_down/hmdb51_org_2'
    
    if not os.path.exists(hmdb51_path):
        print("⚠️ HMDB51 videos not found, using minimal test set")
        return None
    
    # Get test videos (different from training)
    test_videos = []
    for root, dirs, files in os.walk(hmdb51_path):
        for file in files[:50]:  # Limited test set
            if file.endswith(('.avi', '.mp4', '.mov')):
                test_videos.append(os.path.join(root, file))
    
    if test_videos:
        test_dir = '/kaggle/working/test_dataset'
        os.makedirs(test_dir, exist_ok=True)
        
        test_data = {
            "image_names": test_videos,
            "image_labels": [i % 51 for i in range(len(test_videos))]
        }
        
        with open(f'{test_dir}/base.json', 'w') as f:
            json.dump(test_data, f)
        
        print(f"✅ Created test dataset with {len(test_videos)} videos")
        return test_dir
    
    return None

def save_reduce_results(test_results, best_result, output_path, successful_chunks, total_chunks):
    """Lưu kết quả reduce phase"""
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare results
    reduce_results = {
        'reduce_status': 'success',
        'total_chunks': total_chunks,
        'successful_chunks': len(successful_chunks),
        'success_rate': len(successful_chunks) / total_chunks * 100,
        'best_chunk_id': best_result['chunk_id'] if best_result else None,
        'best_accuracy': best_result['accuracy'] if best_result else 0.0,
        'all_test_results': test_results,
        'timestamp': str(torch.tensor(0).item())  # Simple timestamp
    }
    
    # Save results
    with open(f'{output_path}/reduce_results.json', 'w') as f:
        json.dump(reduce_results, f, indent=2)
    
    # Copy best model
    if best_result and os.path.exists(best_result['model_path']):
        import shutil
        shutil.copy2(best_result['model_path'], f'{output_path}/best_model.tar')
        print(f"✅ Copied best model to {output_path}/best_model.tar")
    
    # Upload to HDFS if possible
    hdfs_result_path = '/user/tamt/output/final_reduce_results.json'
    upload_result = run_hdfs_command(['dfs', '-put', '-f', 
                                     f'{output_path}/reduce_results.json', 
                                     hdfs_result_path])
    
    if upload_result.returncode == 0:
        print("✅ Uploaded final results to HDFS")
    
    return reduce_results

def download_trained_models(successful_chunks):
    """Download trained models from successful chunks"""
    trained_models = []
    
    print("📥 Downloading trained models...")
    
    for chunk_id in successful_chunks:
        # Check multiple locations for model files
        model_locations = [
            f'/tmp/map_output_chunk_{chunk_id}/best_model.tar',
            f'/tmp/map_output_chunk_{chunk_id}/checkpoints/hmdb51/VideoMAES_meta_deepbdc_5way_5shot_2TAA/tmp/map_output_chunk_{chunk_id}/best_model.tar',
            f'/tmp/working_chunk_{chunk_id}/checkpoints/hmdb51/best_model.tar',
            f'/tmp/sequential_model_chunk_{chunk_id}.tar',  # For sequential training
            f'/tmp/map_output_chunk_{chunk_id}/best_model_chunk_{chunk_id}.tar'
        ]
        
        model_found = False
        
        for model_path in model_locations:
            if os.path.exists(model_path):
                try:
                    # Get model file size
                    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                    
                    model_info = {
                        'chunk_id': chunk_id,
                        'model_path': model_path,
                        'size': model_size
                    }
                    
                    trained_models.append(model_info)
                    print(f"✅ Found local model for chunk {chunk_id}")
                    print(f"✅ Model {chunk_id}: {model_size:.1f}MB")
                    
                    model_found = True
                    break
                    
                except Exception as e:
                    print(f"⚠️ Error accessing model {model_path}: {e}")
                    continue
        
        if not model_found:
            print(f"❌ No model found for chunk {chunk_id}")
            
            # Try to find any .tar or .pth files in chunk output directory
            chunk_output_dir = f'/tmp/map_output_chunk_{chunk_id}'
            if os.path.exists(chunk_output_dir):
                import glob
                model_files = glob.glob(f'{chunk_output_dir}/**/*.tar', recursive=True)
                model_files.extend(glob.glob(f'{chunk_output_dir}/**/*.pth', recursive=True))
                
                if model_files:
                    # Take the first found model
                    model_path = model_files[0]
                    model_size = os.path.getsize(model_path) / (1024 * 1024)
                    
                    model_info = {
                        'chunk_id': chunk_id,
                        'model_path': model_path,
                        'size': model_size
                    }
                    
                    trained_models.append(model_info)
                    print(f"✅ Found alternative model for chunk {chunk_id}: {os.path.basename(model_path)}")
    
    return trained_models

# Complete fixed reduce_worker.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_chunks', type=int, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    print(f"🔄 REDUCE WORKER STARTING")
    print("=" * 40)
    
    try:
        # Setup
        setup_environment()
        
        # Collect map results with REAL data
        map_results, successful_chunks = collect_map_results(args.num_chunks)
        
        print(f"📊 Successful chunks: {len(successful_chunks)}/{args.num_chunks}")
        
        if not successful_chunks:
            print("❌ No successful chunks found!")
            empty_results = {
                'reduce_status': 'failed',
                'total_chunks': args.num_chunks,
                'successful_chunks': 0,
                'reason': 'No successful map workers'
            }
            os.makedirs(args.output_path, exist_ok=True)
            with open(f'{args.output_path}/reduce_results.json', 'w') as f:
                json.dump(empty_results, f)
            return
        
        # Download trained models - NOW PROPERLY DEFINED
        trained_models = download_trained_models(successful_chunks)
        
        if not trained_models:
            print("❌ No trained models found!")
            # Create empty results for failed case
            empty_results = {
                'reduce_status': 'failed',
                'total_chunks': args.num_chunks,
                'successful_chunks': len(successful_chunks),
                'reason': 'No trained models found'
            }
            os.makedirs(args.output_path, exist_ok=True)
            with open(f'{args.output_path}/reduce_results.json', 'w') as f:
                json.dump(empty_results, f)
            return
        
        # Evaluate all models using REAL results
        test_results = []
        for model_info in trained_models:
            result = evaluate_model_performance(model_info, map_results)
            test_results.append(result)
        
        # Filter out failed evaluations
        valid_results = [r for r in test_results if r['accuracy'] > 0.0]
        
        if not valid_results:
            print("❌ No valid accuracy results found!")
            # Still save what we have
            empty_results = {
                'reduce_status': 'failed',
                'total_chunks': args.num_chunks,
                'successful_chunks': len(successful_chunks),
                'reason': 'No valid accuracy results',
                'failed_results': test_results
            }
            os.makedirs(args.output_path, exist_ok=True)
            with open(f'{args.output_path}/reduce_results.json', 'w') as f:
                json.dump(empty_results, f)
            return
        
        # Select best model based on REAL accuracy
        best_result = select_best_model(valid_results)
        
        # Save results
        final_results = save_reduce_results(valid_results, best_result, args.output_path, 
                                          successful_chunks, args.num_chunks)
        
        print(f"\n🏆 REDUCE PHASE COMPLETED (REAL RESULTS)")
        print(f"   ✅ Success Rate: {final_results['success_rate']:.1f}%")
        print(f"   🎯 Best REAL Accuracy: {final_results['best_accuracy']:.2f}%")
        print(f"   🏆 Best Chunk: {final_results['best_chunk_id']}")
        
        # Show all real results
        print(f"\n📊 All REAL Results:")
        for result in valid_results:
            print(f"   Chunk {result['chunk_id']}: {result['accuracy']:.2f}% (source: {result['source']})")
        
    except Exception as e:
        print(f"💥 Reduce Worker Exception: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error results
        error_results = {
            'reduce_status': 'error',
            'total_chunks': args.num_chunks,
            'error_message': str(e)
        }
        os.makedirs(args.output_path, exist_ok=True)
        with open(f'{args.output_path}/reduce_results.json', 'w') as f:
            json.dump(error_results, f)

if __name__ == '__main__':
    main()
