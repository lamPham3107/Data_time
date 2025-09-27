#!/bin/bash
# filepath: run_mapreduce.sh

# Cấu hình
gpuid=0
DATA_ROOT=/kaggle/input/tamt-bigdata/TAMT/filelist/hmdb51-molo
MODEL_PATH=/kaggle/working/checkpoints/hmdb51/best_model.tar
OUTPUT_PATH=/kaggle/working/checkpoints/hmdb51_meta
HDFS_ROOT=/kaggle/working/hdfs_simulation
NUM_CHUNKS=4

cd /kaggle/input/tamt-bigdata/TAMT

echo "============= Setup HDFS and Split Dataset ============="
python setup_hdfs.py

echo "============= Map Phase: Parallel Training ============="
# Chạy map workers song song
for chunk_id in $(seq 0 $((NUM_CHUNKS-1))); do
    echo "Starting Map Worker $chunk_id..."
    python map_worker.py \
        --chunk_id $chunk_id \
        --hdfs_root $HDFS_ROOT \
        --model_path $MODEL_PATH \
        --gpu_id $gpuid &
done

# Đợi tất cả map workers
wait
echo "All Map workers completed!"

echo "============= Reduce Phase: Model Selection ============="
python reduce_worker.py \
    --hdfs_root $HDFS_ROOT \
    --data_path $DATA_ROOT \
    --output_path $OUTPUT_PATH \
    --num_chunks $NUM_CHUNKS \
    --gpu_id $gpuid

echo "============= Final Test ============="
python test.py --dataset ucf101 --data_path $DATA_ROOT \
    --model VideoMAES --method meta_deepbdc --image_size 112 \
    --gpu $gpuid --n_shot 5 --reduce_dim 256 \
    --model_path $OUTPUT_PATH/best_model.tar --test_task_nums 5

echo "MapReduce Pipeline Completed!"