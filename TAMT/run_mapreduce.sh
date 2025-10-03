#!/bin/bash

# Environment setup
export HADOOP_HOME=/kaggle/working/hadoop
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

# Configuration
NUM_CHUNKS=4
OUTPUT_PATH=/kaggle/working/mapreduce_results
MODE=${1:-sequential}  # Change default to sequential for stability

echo "🚀 LAUNCHING MAPREDUCE TRAINING PIPELINE"
echo "========================================="
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
rm -f /tmp/sequential_model_chunk_*.tar
rm -rf /tmp/chunk_*_dataset
rm -rf /tmp/map_output_chunk_*
rm -rf /tmp/working_chunk_*
rm -rf $OUTPUT_PATH

echo ""
echo "=== MAP PHASE: Real Video Training ==="

# Fix timeout cho sequential mode:
if [ "$MODE" = "sequential" ]; then
    # Sequential mode với proper timeouts
    for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
        echo ""
        echo "🔥 Map Worker $chunk_id (Sequential)..."
        
        # DIFFERENT TIMEOUTS per chunk based on complexity
        if [ $chunk_id -eq 0 ]; then
            timeout 3000 python map_worker.py --chunk_id $chunk_id --gpu_id 0  # 50 minutes for chunk 0
        elif [ $chunk_id -eq 1 ]; then
            timeout 2700 python map_worker.py --chunk_id $chunk_id --gpu_id 0  # 45 minutes for chunk 1
        elif [ $chunk_id -eq 2 ]; then
            timeout 2400 python map_worker.py --chunk_id $chunk_id --gpu_id 0  # 40 minutes for chunk 2
        else
            timeout 2100 python map_worker.py --chunk_id $chunk_id --gpu_id 0  # 35 minutes for chunk 3
        fi
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ Worker $chunk_id completed successfully"
        elif [ $exit_code -eq 124 ]; then
            echo "⏰ Worker $chunk_id timed out"
        else
            echo "❌ Worker $chunk_id failed (exit code: $exit_code)"
        fi
        
        # Longer cleanup for stability
        if [ $chunk_id -lt $((NUM_CHUNKS-1)) ]; then
            echo "⏳ Waiting 15 seconds for GPU cleanup..."
            sleep 15
        fi
    done
    
else
    # Parallel mode fix:
    echo "🚀 Starting all workers in parallel..."
    
    for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
        echo "🔥 Starting Map Worker $chunk_id..."
        
        # BACKGROUND PROCESSES with proper timeouts
        if [ $chunk_id -eq 0 ]; then
            timeout 3000 python map_worker.py --chunk_id $chunk_id --gpu_id 0 &  # 50 minutes
        elif [ $chunk_id -eq 1 ]; then
            timeout 2700 python map_worker.py --chunk_id $chunk_id --gpu_id 0 &  # 45 minutes
        elif [ $chunk_id -eq 2 ]; then
            timeout 2400 python map_worker.py --chunk_id $chunk_id --gpu_id 0 &  # 40 minutes
        else
            timeout 2100 python map_worker.py --chunk_id $chunk_id --gpu_id 0 &  # 35 minutes
        fi
        
        # Stagger worker starts
        if [ $chunk_id -lt $((NUM_CHUNKS-1)) ]; then
            sleep 10  # Longer stagger
        fi
    done
    
    echo "⏳ Waiting for all workers to complete..."
    wait
    echo "✅ All Map workers completed!"
fi

echo ""
echo "=== REDUCE PHASE: Model Selection ==="
python reduce_worker.py \
    --num_chunks $NUM_CHUNKS \
    --output_path $OUTPUT_PATH \
    --gpu_id 0

echo ""
echo "=== FINAL RESULTS ==="
if [ -f "$OUTPUT_PATH/reduce_results.json" ]; then
    echo "📊 Results saved to: $OUTPUT_PATH/reduce_results.json"
    
    # Show summary with better error handling
    python3 -c "
import json
import sys
try:
    with open('$OUTPUT_PATH/reduce_results.json', 'r') as f:
        results = json.load(f)
    print(f'🏆 Success Rate: {results.get(\"success_rate\", 0):.1f}%')
    print(f'🎯 Best Accuracy: {results.get(\"best_accuracy\", 0):.2f}%')
    print(f'📈 Successful Chunks: {results.get(\"successful_chunks\", 0)}/{results.get(\"total_chunks\", 4)}')
    
    # Show individual results
    if 'all_test_results' in results:
        print('\\n📊 Individual Results:')
        for result in results['all_test_results']:
            chunk_id = result.get('chunk_id', 'N/A')
            accuracy = result.get('accuracy', 0)
            source = result.get('source', 'unknown')
            print(f'   Chunk {chunk_id}: {accuracy:.2f}% (source: {source})')
            
except Exception as e:
    print(f'❌ Error reading results: {e}')
    sys.exit(1)
"
else
    echo "❌ No results file found"
fi

echo ""
echo "🎉 MAPREDUCE PIPELINE COMPLETED!"

# Show system status
echo ""
echo "📊 MAPREDUCE EXECUTION COMPLETED"