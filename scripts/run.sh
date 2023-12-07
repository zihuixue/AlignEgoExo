
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset-name>"
    exit 1
fi

if [ "$1" = "break_eggs" ]; then
  python train.py --dataset break_eggs \
                --task align_bbox --sample_by_bbox \
                --hidden_dim 256 --n_layers 1  \
                --loss dtw_contrastive --dtw_shuffle_num 30 --dtw_ratio 1 \
                --output_dir bestconfig


elif [ "$1" = "pour_milk" ]; then
  # pour milk, missing one det_bounding_box.pickle file (to be updated later), basic no object-centric encoder version below
  python train.py --dataset pour_milk \
                --task align \
                --lr 1e-4 \
                --hidden_dim 256 --n_layers 1  \
                --loss dtw \
                --output_dir base_version

elif [ "$1" = "pour_liquid" ]; then
  python train.py --dataset pour_liquid \
                --task align_bbox --sample_by_bbox \
                --use_bbox_pe --weigh_token_by_bbox \
                --hidden_dim 128 --n_layers 3  \
                --loss dtw_contrastive --dtw_shuffle_num 16 --dtw_ratio 2 \
                --output_dir bestconfig

elif [ "$1" = "tennis_forehand" ]; then
  python train.py --dataset tennis_forehand \
                --num_frames 20 --ds_every_n_epoch 1 \
                --task align_bbox --sample_by_bbox \
                --use_bbox_pe --weigh_token_by_bbox --use_mask --one_object_bbox \
                --hidden_dim 128 --n_layers 1  \
                --loss dtw_contrastive --dtw_shuffle_num 10 --dtw_ratio 4 \
                --output_dir bestconfig
else
    echo "Unknown dataset: $1, select among [break_eggs, pour_milk, pour_liquid, tennis_forehand]"
    exit 2
fi