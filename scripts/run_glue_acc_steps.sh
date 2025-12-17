python run_glue.py --model_name_or_path roberta-base --enable_galore --lora_all_modules --max_length 512 --seed=1234 --lora_r 4 --galore_scale 4 --learning_rate 3e-5 --num_train_epochs 30 --task_name mrpc --output_dir results/ft/roberta_base_mrpc_T10/mrpc --update_proj_gap 10 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 8
python run_glue.py --model_name_or_path roberta-base --enable_galore --lora_all_modules --max_length 512 --seed=1234 --lora_r 4 --galore_scale 4 --learning_rate 3e-5 --num_train_epochs 30 --task_name mrpc --output_dir results/ft/roberta_base_mrpc_T10/mrpc --update_proj_gap 10 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2
python run_glue.py --model_name_or_path roberta-base --enable_galore --lora_all_modules --max_length 512 --seed=1234 --lora_r 4 --galore_scale 4 --learning_rate 3e-5 --num_train_epochs 30 --task_name mrpc --output_dir results/ft/roberta_base_mrpc_T10/mrpc --update_proj_gap 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1
python run_glue.py --model_name_or_path roberta-base --enable_galore --lora_all_modules --max_length 512 --seed=1234 --lora_r 4 --galore_scale 4 --learning_rate 3e-5 --num_train_epochs 30 --task_name mrpc --output_dir results/ft/roberta_base_mrpc_T10/mrpc --update_proj_gap 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1







