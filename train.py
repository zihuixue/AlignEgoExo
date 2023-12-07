import os
import json

from pytorch_lightning import Trainer
from utils.config import argparser
from utils.util import CustomModelCheckpoint
from video_tasks import VideoAlignment, ObjectVideoAlignment


def main():
    task = ObjectVideoAlignment(args) if 'bbox' in args.task else VideoAlignment(args)

    custom_checkpoint_callback = CustomModelCheckpoint(
        every_n_epochs=args.save_every,
        filename="{epoch}",
        save_top_k=-1,
    )

    trainer = Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        callbacks=custom_checkpoint_callback,
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
    )

    if args.eval_only:
        # trainer.validate(task, ckpt_path=args.ckpt)
        trainer.test(task, ckpt_path=args.ckpt)
    else:
        trainer.fit(task)


if __name__ == '__main__':
    args = argparser.parse_args()
    args.output_dir = os.path.join('./logs/exp_'+args.dataset, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    main()