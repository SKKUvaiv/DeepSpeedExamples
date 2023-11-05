# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT
rm -rf /tmp/data_files/
Num_Padding_at_Beginning=0 # this is model related

Actor_Lr=5e-4
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --data_path local/jsonfile \
   --data_split 0,0,10 \
   --actor_model_name_or_path jojo0217/SFT_12.8B_mk2_v2 \
   --critic_model_name_or_path jojo0217/step2_v2_mk1 \
   --num_padding_at_beginning 0 \
   --per_device_training_batch_size 8 \
   --per_device_generation_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 16 \
   --num_warmup_steps 10 \
   --deepspeed --seed 1234 \
   --actor_gradient_checkpointing \
   --offload_reference_model \
   --actor_lora_dim 8 \
   --critic_lora_dim 8 \
   --actor_lora_module_name query_key_value \
   --critic_lora_module_name query_key_value \
   --only_optimize_lora \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --print_answers \
   --print_answers_interval 5 \
   --output_dir $OUTPUT 
