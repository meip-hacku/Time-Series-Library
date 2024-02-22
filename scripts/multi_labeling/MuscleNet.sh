export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name multi_labeling \
  --is_training 1 \
  --root_path ./dataset/SquatDataset/ \
  --model_id SquatTraining \
  --model MuscleNet \
  --data Squat \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
  --embed fixed