mapreduce_script = '''#!/bin/bash

# Environment setup
export HADOOP_HOME=/kaggle/working/hadoop
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

# Configuration
NUM_CHUNKS=4
OUTPUT_PATH=/kaggle/working/mapreduce_results
MODE=${1:-parallel}  # parallel or sequential

echo "🚀 REAL MAPREDUCE WITH VIDEO DATA"
echo "=================================="
echo "Mode: $MODE"
echo "Chunks: $NUM_CHUNKS"
echo "Output: $OUTPUT_PATH"

cd /kaggle/working

# Cleanup
echo "🧹 Cleaning previous runs..."
rm -f /tmp/chunk_*.json
rm -f /tmp/pretrained_model*.tar
rm -f /tmp/map_result_*.json
rm -f /tmp/model_chunk_*.tar
rm -rf /tmp/chunk_*_dataset
rm -rf /tmp/map_output_chunk_*
rm -rf $OUTPUT_PATH

echo ""
echo "=== MAP PHASE: Real Video Training ==="

if [ "$MODE" = "sequential" ]; then
    # Sequential mode - safer for resource constraints
    for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
        echo ""
        echo "🔥 Map Worker $chunk_id (Sequential)..."
        python map_worker.py --chunk_id $chunk_id --gpu_id 0
        
        if [ $? -eq 0 ]; then
            echo "✅ Worker $chunk_id completed successfully"
        else
            echo "❌ Worker $chunk_id failed"
        fi
    done
else
    # Parallel mode - faster but more resource intensive
    echo "Starting all workers in parallel..."
    for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
        echo "🔥 Starting Map Worker $chunk_id..."
        python map_worker.py --chunk_id $chunk_id --gpu_id 0 &
    done
    
    echo "⏳ Waiting for all workers to complete..."
    wait
    echo "✅ All Map workers completed!"
fi

echo ""
echo "=== REDUCE PHASE: Model Selection ==="
python reduce_worker.py --num_chunks 4 --output_path /kaggle/working/mapreduce_results --gpu_id 0

echo ""
echo "=== FINAL RESULTS ==="
if [ -f "$OUTPUT_PATH/reduce_results.json" ]; then
    echo "📊 Results saved to: $OUTPUT_PATH/reduce_results.json"
    
    # Show summary
    python -c "
import json
try:
    with open('$OUTPUT_PATH/reduce_results.json', 'r') as f:
        results = json.load(f)
    print(f'🏆 Success Rate: {results.get(\"success_rate\", 0):.1f}%')
    print(f'🎯 Best Accuracy: {results.get(\"best_accuracy\", 0):.2f}%')
    print(f'📈 Successful Chunks: {results.get(\"successful_chunks\", 0)}/{results.get(\"total_chunks\", 4)}')
except:
    print('❌ Could not read results')
"
else
    echo "❌ No results file found"
fi

echo ""
echo "🎉 MAPREDUCE PIPELINE COMPLETED!"
'''

with open('/kaggle/working/run_mapreduce.sh', 'w') as f:
    f.write(mapreduce_script)

os.chmod('/kaggle/working/run_mapreduce.sh', 0o755)
print("✅ Created run_mapreduce.sh")mapreduce_script = '''#!/bin/bash

# Environment setup
export HADOOP_HOME=/kaggle/working/hadoop
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

# Configuration
NUM_CHUNKS=4
OUTPUT_PATH=/kaggle/working/mapreduce_results
MODE=${1:-parallel}  # parallel or sequential

echo "🚀 REAL MAPREDUCE WITH VIDEO DATA"
echo "=================================="
echo "Mode: $MODE"
echo "Chunks: $NUM_CHUNKS"
echo "Output: $OUTPUT_PATH"

cd /kaggle/working

# Cleanup
echo "🧹 Cleaning previous runs..."
rm -f /tmp/chunk_*.json
rm -f /tmp/pretrained_model*.tar
rm -f /tmp/map_result_*.json
rm -f /tmp/model_chunk_*.tar
rm -rf /tmp/chunk_*_dataset
rm -rf /tmp/map_output_chunk_*
rm -rf $OUTPUT_PATH

echo ""
echo "=== MAP PHASE: Real Video Training ==="

if [ "$MODE" = "sequential" ]; then
    # Sequential mode - safer for resource constraints
    for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
        echo ""
        echo "🔥 Map Worker $chunk_id (Sequential)..."
        python map_worker.py --chunk_id $chunk_id --gpu_id 0
        
        if [ $? -eq 0 ]; then
            echo "✅ Worker $chunk_id completed successfully"
        else
            echo "❌ Worker $chunk_id failed"
        fi
    done
else
    # Parallel mode - faster but more resource intensive
    echo "Starting all workers in parallel..."
    for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
        echo "🔥 Starting Map Worker $chunk_id..."
        python map_worker.py --chunk_id $chunk_id --gpu_id 0 &
    done
    
    echo "⏳ Waiting for all workers to complete..."
    wait
    echo "✅ All Map workers completed!"
fi

echo ""
echo "=== REDUCE PHASE: Model Selection ==="
python reduce_worker.py \\
    --num_chunks $NUM_CHUNKS \\
    --output_path $OUTPUT_PATH \\
    --gpu_id 0

echo ""
echo "=== FINAL RESULTS ==="
if [ -f "$OUTPUT_PATH/reduce_results.json" ]; then
    echo "📊 Results saved to: $OUTPUT_PATH/reduce_results.json"
    
    # Show summary
    python -c "
import json
try:
    with open('$OUTPUT_PATH/reduce_results.json', 'r') as f:
        results = json.load(f)
    print(f'🏆 Success Rate: {results.get(\"success_rate\", 0):.1f}%')
    print(f'🎯 Best Accuracy: {results.get(\"best_accuracy\", 0):.2f}%')
    print(f'📈 Successful Chunks: {results.get(\"successful_chunks\", 0)}/{results.get(\"total_chunks\", 4)}')
except:
    print('❌ Could not read results')
"
else
    echo "❌ No results file found"
fi

echo ""
echo "🎉 MAPREDUCE PIPELINE COMPLETED!"
'''

with open('/kaggle/working/run_mapreduce.sh', 'w') as f:
    f.write(mapreduce_script)

os.chmod('/kaggle/working/run_mapreduce.sh', 0o755)
print("✅ Created run_mapreduce.sh")