import pickle
import glob
import subprocess
import os
import argparse

def collect_map_results(hdfs_root, num_chunks):
    """Thu thập kết quả từ map workers"""
    map_results = []
    successful_chunks = []
    
    for i in range(num_chunks):
        result_file = f"{hdfs_root}/map_output/map_result_{i}.pkl"
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
                map_results.append(result)
                
                if result['status'] == 'completed':
                    successful_chunks.append(i)
                    print(f"Reduce: Collected successful result from chunk {i}")
                else:
                    print(f"Reduce: Chunk {i} failed - {result['stderr'][:200]}")
    
    return map_results, successful_chunks

def find_trained_models(hdfs_root, successful_chunks):
    """Tìm các model đã train thành công"""
    trained_models = []
    
    for chunk_id in successful_chunks:
        model_dir = f"{hdfs_root}/map_output/chunk_{chunk_id}_model"
        best_model_path = f"{model_dir}/best_model.tar"
        
        if os.path.exists(best_model_path):
            trained_models.append({
                'chunk_id': chunk_id,
                'model_path': best_model_path
            })
            print(f"Reduce: Found trained model from chunk {chunk_id}")
    
    return trained_models

def test_model(model_info, data_path, gpu_id=0):
    """Test một model và trả về accuracy"""
    cmd = [
        'python', 'test.py',
        '--dataset', 'ucf101',
        '--data_path', data_path,
        '--model', 'VideoMAES',
        '--method', 'meta_deepbdc',
        '--image_size', '112',
        '--gpu', str(gpu_id),
        '--n_shot', '5',
        '--reduce_dim', '256',
        '--model_path', model_info['model_path'],
        '--test_task_nums', '5'
    ]
    
    print(f"Testing model from chunk {model_info['chunk_id']}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse accuracy từ output
    accuracy = 0.0
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'Test Acc' in line and '%' in line:
                try:
                    acc_str = line.split('=')[1].split('%')[0].strip()
                    accuracy = float(acc_str)
                    break
                except:
                    continue
    
    return accuracy, result.stdout

def select_best_model(trained_models, data_path, gpu_id=0):
    """Test tất cả models và chọn best"""
    best_accuracy = 0.0
    best_model = None
    test_results = []
    
    for model_info in trained_models:
        accuracy, output = test_model(model_info, data_path, gpu_id)
        
        test_result = {
            'chunk_id': model_info['chunk_id'],
            'model_path': model_info['model_path'],
            'accuracy': accuracy,
            'output': output
        }
        test_results.append(test_result)
        
        print(f"Model from chunk {model_info['chunk_id']}: {accuracy:.2f}% accuracy")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_info
    
    return best_model, best_accuracy, test_results

def save_final_result(hdfs_root, best_model, best_accuracy, map_results, test_results):
    """Lưu kết quả cuối cùng"""
    final_result = {
        'best_accuracy': best_accuracy,
        'best_model_path': best_model['model_path'] if best_model else None,
        'best_chunk_id': best_model['chunk_id'] if best_model else None,
        'num_chunks_processed': len([r for r in map_results if r['status'] == 'completed']),
        'map_results': map_results,
        'test_results': test_results
    }
    
    with open(f"{hdfs_root}/reduce_output/final_result.pkl", 'wb') as f:
        pickle.dump(final_result, f)
    
    return final_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_root', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--num_chunks', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    print("Starting Reduce Phase...")
    
    # Thu thập kết quả map
    map_results, successful_chunks = collect_map_results(args.hdfs_root, args.num_chunks)
    
    # Tìm models đã train
    trained_models = find_trained_models(args.hdfs_root, successful_chunks)
    
    if not trained_models:
        print("No trained models found!")
        return
    
    # Chọn best model
    best_model, best_accuracy, test_results = select_best_model(trained_models, args.data_path, args.gpu_id)
    
    # Lưu kết quả
    final_result = save_final_result(args.hdfs_root, best_model, best_accuracy, map_results, test_results)
    
    # Copy best model to output
    if best_model:
        import shutil
        os.makedirs(args.output_path, exist_ok=True)
        shutil.copy2(best_model['model_path'], f"{args.output_path}/best_model.tar")
        
        print("=" * 50)
        print("MapReduce Reduce Phase Completed!")
        print(f"Best Model Accuracy: {best_accuracy:.2f}%")
        print(f"Best Model from Chunk: {best_model['chunk_id']}")
        print(f"Final model copied to: {args.output_path}/best_model.tar")
        print("=" * 50)

if __name__ == '__main__':
    main()