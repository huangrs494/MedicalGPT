python pretraining.py \
    --model_type bloom \
    --model_name_or_path /root/.cache/modelscope/hub/AI-ModelScope/bloomz-7b1 \
    --train_file_dir ./data/pretrain_v1 \
    --validation_file_dir ./data/pretrain_v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 10000 \
    --max_eval_samples 10 \
    --num_train_epochs 8 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 10 \
    --block_size 512 \
    --group_by_length True \
    --output_dir models/output/outputs-pt-bloomz-7b1-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 12 \
    --lora_alpha 24 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache
