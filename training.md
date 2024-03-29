deepspeed fastchat/train/train_lora.py \
    --deepspeed ./zero3.json \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path lmsys/vicuna-7b-v1.5-16k \
    --fp16 True \
    --output_dir ./output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --evaL_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True
    --q_lora False

deepspeed fastchat/train/train_lora.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path /home/devonperoutky/dataset/advice/augmented/full_dataset.json \
    --output_dir ./checkpoints \
    --num_train_epochs 2 \
    --fp16 True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora False \
    --gradient_checkpointing True \
    --flash_attn False