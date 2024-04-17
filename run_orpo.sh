CUDA_VISIBLE_DEVICES=0,1 python orpo_training.py \
    --model_type auto \
    --model_name_or_path Qwen/Qwen1.5-0.5B-Chat \
    --train_file_dir ./data/reward \
    --validation_file_dir ./data/reward \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --max_steps 100 \
    --eval_steps 20 \
    --save_steps 50 \
    --max_source_length 128 \
    --max_target_length 128 \
    --output_dir outputs-orpo-qwen-v1 \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --fp16 True \
    --device_map auto \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --orpo_beta 0.1 \
    --cache_dir ./cache
