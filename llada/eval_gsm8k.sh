# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TORCHDYNAMO_DISABLE=1

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
# You can change the model path to LLaDA-1.5 by setting model_path='GSAI-ML/LLaDA-1.5'


# baseline # 48h on 1 A100
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_smc=False,show_speed=True 

# cache+parallel
accelerate launch --main_process_port 29503 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,use_smc=True,show_speed=True 



# --------
### haven't finished debug !!! ###
# prefix cache+parallel factor # 1h on 1 A100 -> 75.89 for non-smc
# accelerate launch --gpu_ids "0" eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,factor=${factor},use_smc=True,show_speed=True
### haven't finished debug !!! ###










# baseline
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True 


# prefix cache
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True 


# parallel
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True

# parallel factor
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},show_speed=True


# prefix cache+parallel
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True

# dual cache+parallel
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

# prefix cache+parallel factor
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,factor=${factor},show_speed=True

# dual cache+parallel factor
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,factor=${factor},show_speed=True
