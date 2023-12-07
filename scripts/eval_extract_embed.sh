ckpt_dir=/checkpoint/sherryxue/AE2/AE2_ckpts

# extract embeddings then evaluate
# break eggs
python evaluation/evaluate_features.py --dataset break_eggs --task align_bbox \
    --extract_embedding \
    --hidden_dim 256 --n_layers 1 \
    --eval_task 1234 \
    --ckpt $ckpt_dir/break_eggs.ckpt

# pour milk, missing one det_bounding_box.pickle file, to be updated later
#python evaluation/evalulate_features.py --dataset pour_milk --task align_bbox \
#    --extract_embedding \
#    --hidden_dim 256 --n_layers 1 \
#    --eval_task 1234 \
#    --ckpt $ckpt_dir/pour_milk.ckpt

# pour liquid
python evaluation/evaluate_features.py --dataset pour_liquid --task align_bbox \
    --extract_embedding \
    --use_bbox_pe --weigh_token_by_bbox \
    --hidden_dim 128 --n_layers 3 \
    --eval_task 1234 \
    --ckpt $ckpt_dir/pour_liquid.ckpt

# tennis forehand
python evaluation/evaluate_features.py --dataset tennis_forehand --task align_bbox \
    --use_bbox_pe --weigh_token_by_bbox --use_mask --one_object_bbox \
    --hidden_dim 128 --n_layers 1 \
    --eval_task 1234 \
    --ckpt $ckpt_dir/tennis_forehand.ckpt
