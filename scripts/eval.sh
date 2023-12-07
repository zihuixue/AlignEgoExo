ckpt_dir=/data_path/AE2/AE2_ckpts

# use extracted embeddings (saved in ckpt_dir/$dataset_eval)
for dataset in break_eggs pour_milk pour_liquid tennis_forehand; do
    python evaluation/evaluate_features.py --dataset $dataset \
        --eval_task 1234 \
        --ckpt $ckpt_dir/$dataset.ckpt
done