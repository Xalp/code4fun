# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=256
block_length=32
steps=$((length / block_length))
model_path='GSAI-ML/LLaDA-8B-Instruct'
factor=1.0
logname=smc-b${block_length}


accelerate launch --main_process_port 29501 eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,use_smc=True,show_speed=True \
--output_path evals_results/cache_parallel/humaneval-ns0-${length}-${logname} --log_samples


#accelerate launch --main_process_port 29502 eval_llada.py --tasks ${task} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=256,block_length=${block_length},use_cache=True,use_smc=True,show_speed=True \
#--output_path evals_results/prefix_cache/humaneval-ns0-${length}-${logname} --log_samples

# python postprocess_code.py evals_results/prefix_cache/humaneval-ns0-256/GSAI-ML__LLaDA-8B-Instruct/samples_humaneval_2026-01-09T22-31-20.967742.jsonl


# baseline
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
# --output_path evals_results/baseline/humaneval-ns0-${length} --log_samples

# prefix cache
#accelerate launch eval_llada.py --tasks ${task} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True \
#--output_path evals_results/prefix_cache/humaneval-ns0-${length} --log_samples

# parallel
#accelerate launch eval_llada.py --tasks ${task} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True \
#--output_path evals_results/parallel/humaneval-ns0-${length} --log_samples

# prefix cache+parallel
#accelerate launch eval_llada.py --tasks ${task} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True \
#--output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples

# dual cache+parallel
#accelerate launch eval_llada.py --tasks ${task} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
#--output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}
