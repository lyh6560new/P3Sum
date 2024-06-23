#finetune summarization task from the checkpoint of roberta-based tuned with ul2-variable on the length=256.
#cd "/data3/whr/lyh/ControllingPoliticalBias/simplex-diffusion-main/"
cd "path/to/simplex-diffusion-main"
model_name="roberta-base"
model_path="model/"
learning_rate=3e-5
max_steps=200000 #120000
CUDA_VISIBLE_DEVICES=1
datasetname="Sampled_Datasets/cnn_dm_500_raw"
dataset_config_name="3.0.0"
cache_dir="./"
preprocessing_num_workers=16
overwrite_cache=false
per_device_train_batch_size=8
per_device_eval_batch_size=16
do_train=false
do_eval=false
do_predict=true
evaluation_strategy="no"
eval_steps=1000
report_to="tensorboard"
overwrite_output_dir=false
max_seq_length=512
max_target_length=100
max_source_length=412
val_max_target_length=100
skip_special_tokens=true
max_eval_samples=100
max_predict_samples=500
simplex_value=5
num_diffusion_steps=1000
lr_scheduler_type="linear"
pad_to_max_length=true
beta_schedule="squaredcos_improved_ddpm"
weight_decay=0.0
warmup_steps=2000
max_steps=200000
gradient_accumulation_steps=1
logging_steps=50
save_steps=20000
conditional_generation="ul2"
save_total_limit=1
tokenized_data_path="raw/xsum/roberta"
metric_for_best_model="rouge1"
if_control=true
ctr_model_name='POLITICS_model' 
ctr_opt_label_idx=0 
output_dir="out/"
decode_ctr_lr=1000


CUDA_VISIBLE_DEVICES='0' python sdlm/run_summarization.py \
--model_name_or_path ${model_path} \
--output_dir ${output_dir} \
--dataset_name ${datasetname} \
--dataset_config_name ${dataset_config_name} \
--cache_dir ${cache_dir} \
--preprocessing_num_workers ${preprocessing_num_workers} \
--overwrite_cache ${overwrite_cache} \
--per_device_train_batch_size ${per_device_train_batch_size} \
--per_device_eval_batch_size ${per_device_eval_batch_size} \
--do_train ${do_train} \
--do_eval ${do_eval} \
--do_predict ${do_predict} \
--evaluation_strategy ${evaluation_strategy} \
--eval_steps ${eval_steps} \
--report_to ${report_to} \
--overwrite_output_dir ${overwrite_output_dir} \
--max_seq_length ${max_seq_length} \
--max_target_length ${max_target_length} \
--max_source_length ${max_source_length} \
--val_max_target_length ${val_max_target_length} \
--skip_special_tokens ${skip_special_tokens} \
--max_eval_samples ${max_eval_samples} \
--max_predict_samples ${max_predict_samples} \
--simplex_value ${simplex_value} \
--num_diffusion_steps ${num_diffusion_steps} \
--lr_scheduler_type ${lr_scheduler_type} \
--pad_to_max_length ${pad_to_max_length} \
--beta_schedule ${beta_schedule} \
--weight_decay ${weight_decay} \
--warmup_steps ${warmup_steps} \
--max_steps ${max_steps} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--logging_steps ${logging_steps} \
--save_steps ${save_steps} \
--conditional_generation ${conditional_generation} \
--save_total_limit ${save_total_limit} \
--tokenized_data_path ${tokenized_data_path} \
--metric_for_best_model ${metric_for_best_model} \
--if_control ${if_control} \
--ctr_model_name ${ctr_model_name} \
--ctr_opt_label_idx ${ctr_opt_label_idx} \
--decode_ctr_lr ${decode_ctr_lr} \
--self_condition "logits_mean" \
--self_condition_mix_before_weights true 

