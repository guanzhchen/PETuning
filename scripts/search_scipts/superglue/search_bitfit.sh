export TASK_NAME=superglue
export epoch=50
export DATASET_NAME=$1

for lr in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
do
    for bs in 16 32
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
        --output_dir checkpoints/$DATASET_NAME-roberta-searchbitfit/$DATASET_NAME-$bs-$lr/ \
        --overwrite_output_dir \
        --hidden_dropout_prob 0.1 \
        --seed 1111 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --load_best_model_at_end\
        --metric_for_best_model loss\
        --greater_is_better False \
        --bitfit
      done
done

python3 ./scripts/search_scipts/search.py $DATASET_NAME roberta bitfit