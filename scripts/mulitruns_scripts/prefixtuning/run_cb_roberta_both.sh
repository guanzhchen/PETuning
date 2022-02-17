export TASK_NAME=superglue
export DATASET_NAME=cb

bs=32
lr=5e-3
dropout=0.1
epoch=50

for model_seed in 1111 2222 3333 4444 5555 6666 7777 8888 9999 101010 111111 222222 333333 444444 555555 666666 777777 888888 999999 10101010
do
  python3 run.py \
    --model_name_or_path roberta-base \
    --task_name $TASK_NAME \
    --dataset_name $DATASET_NAME \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --output_dir both_seeds/$DATASET_NAME-roberta-pt/$seed-$model_seed/ \
    --overwrite_output_dir \
    --hidden_dropout_prob $dropout \
    --seed $model_seed \
    --model_seed $model_seed \
    --save_strategy epoch \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --load_best_model_at_end\
    --metric_for_best_model loss\
    --greater_is_better False \
    --prefix \
    --pre_seq_len 8
done