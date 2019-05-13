export GLUE_DIR=/home/echowdh2/Research_work/processed_multimodal_data/
export CACHE_DIR=/home/echowdh2/Research_work/processed_multimodal_data/MRPC/model_cache

export TASK_NAME=MRPC

python run_classifier.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --cache_dir $CACHE_DIR \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --output_dir /tmp/$TASK_NAME/
