import argparse

argparser = argparse.ArgumentParser(description='Align Ego Exo')
# Training
argparser.add_argument('--num_gpus', type=int, default=2, help='gpus')
argparser.add_argument('--task', type=str, default='align', help='Tasks: align or align_bbox')
argparser.add_argument('--eval_only', action='store_true', help='eval only')
argparser.add_argument('--output_dir', type=str, default='debug', help='Path to results')
argparser.add_argument('--ds_every_n_epoch', type=int, default=10, help='downstream evaluation every n epochs')
argparser.add_argument('--save_every', type=int, default=10, help='save ckpt every n epochs')
argparser.add_argument('--epochs', type=int, default=300, help='Maximum epoch')
argparser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
argparser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
argparser.add_argument('--batch_size', type=int, default=2, help='batch size')
argparser.add_argument('--num_workers', type=int, default=0, help='number of workers')

# Data
argparser.add_argument('--dataset_root', type=str, default='/data_path/AE2/AE2_data', help='dataset root')
argparser.add_argument('--dataset', type=str, default='break_eggs', help='dataset name')
argparser.add_argument('--view1', type=str, default='ego', help='view 1')
argparser.add_argument('--view2', type=str, default='exo', help='view 2 (can be same as view 1)')
argparser.add_argument('--input_size', type=int, default=224, help='frame input size: 168 or 224')
argparser.add_argument('--num_frames', type=int, default=32, help='number of frames: 20 or 32')
argparser.add_argument('--frame_stride', type=int, default=15, help='frame stride')
argparser.add_argument('--num_context_steps', type=int, default=2, help='context steps')
argparser.add_argument('--random_offset', type=int, default=1, help='random offset')
# Data (object bounding box)
argparser.add_argument('--bbox_expansion_ratio', type=float, default=1.0, help='Bounding box expansion ratio')
argparser.add_argument('--bbox_threshold', type=float, default=0.0, help='Bounding box threshold')
argparser.add_argument('--one_object_bbox', action='store_true', help='use one object bbox, not two objects')
argparser.add_argument('--sample_by_bbox', action='store_true', help='(tennis) sample frames by bbox')

# Model
argparser.add_argument('--base_model_name', type=str, default='resnet50', help='Base model name')
argparser.add_argument('--freeze_base', action='store_true', help='whether to freeze base model')
argparser.add_argument('--hidden_dim', type=int, default=128, help='transformer hidden dim')
argparser.add_argument('--n_layers', type=int, default=3, help='transformer layer num')
argparser.add_argument('--embedding_size', type=int, default=128, help='output embedding size')
# Model (object bounding box)
argparser.add_argument('--use_mask', action='store_true', help='transformer use mask')
argparser.add_argument('--use_bbox_pe', action='store_true', help='use bounding box positional embedding')
argparser.add_argument('--weigh_token_by_bbox', action='store_true', help='whether to weigh local tokens by bbox confidence')

# Alignment loss
argparser.add_argument('--loss', type=str, default='dtw', help='train loss: tcc/dtw/vava')
argparser.add_argument('--tcc_temp', type=float, default=0.1, help='if loss is TCC, temperature')
argparser.add_argument('--dtw_scale_factor', type=float, default=0.01, help='DTW loss scale factor')
argparser.add_argument('--dtw_beta', type=float, default=0.0, help='DTW contrastive beta')
argparser.add_argument('--dtw_ratio', type=float, default=1.0, help='DTW contrastive loss ratio')
argparser.add_argument('--dtw_shuffle_num', type=int, default=4, help='DTW contrastive loss shuffle num')

# Eval
argparser.add_argument('--ckpt', type=str, default='', help='model ckpt')
argparser.add_argument('--extract_embedding', action='store_true', help='extract embeddings')
argparser.add_argument('--eval_task', type=str, default='1234', help='downstream evaluation')
argparser.add_argument('--eval_mode', type=str, default='test', help='evaluation time use val/test split')
