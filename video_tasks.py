import copy
import os
import numpy as np
import torch
from pytorch_lightning.core import LightningModule
from dataset.video_align_dataset import VideoAlignmentTrainDataset
from dataset.video_align_dataset_bbox import VideoAlignmentBboxTrainDataset
from models.embedder import Embedder, RoIPosEmbedder
from models.loss import AlignmentLoss
from evaluation.evaluate_features import prepare_data_loader, extract_embedding, classification, frame_retrieval, compute_progression_value, kendalls_tau


class VideoAlignment(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Embedder(args)
        self.loss = AlignmentLoss(args)
        self.checkpoint_metric = "train_loss"
        self.data_path = None
        self.object_box = True if 'bbox' in args.task else False
        self.ds_loader_train, self.ds_dataset_train = prepare_data_loader(args, 'train', batch_size=256, bbox=self.object_box)
        self.ds_loader_val, self.ds_dataset_val = prepare_data_loader(args, 'val', batch_size=256, bbox=self.object_box)
        print(f'constructing downstream loader train {len(self.ds_loader_train)} | val {len(self.ds_loader_val)}')

    def training_step(self, batch, batch_idx):
        frames, steps, seq_lens = batch
        x1 = frames[:, 0, ...].permute(0, 1, 4, 2, 3)  # (bs, 64, 3, 168, 168)
        x2 = frames[:, 1, ...].permute(0, 1, 4, 2, 3)

        embeds1 = self.model(x1)
        embeds2 = self.model(x2)
        embeddings = torch.stack((embeds1, embeds2), dim=1)  # (bs, 2, 32, 128)

        loss, loss2 = self.loss(embeddings, steps, seq_lens, self.global_step)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_loss_reg', loss2, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, steps, seq_lens = batch
        x1 = frames[:, 0, ...].permute(0, 1, 4, 2, 3)  # (bs, 64, 3, 168, 168)
        x2 = frames[:, 1, ...].permute(0, 1, 4, 2, 3)

        embeds1 = self.model(x1)
        embeds2 = self.model(x2)
        embeddings = torch.stack((embeds1, embeds2), dim=1)  # (bs, 2, 32, 128)

        loss, loss2 = self.loss(embeddings, steps, seq_lens, self.global_step)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_loss_reg', loss2, on_step=True, on_epoch=True)

        self.evaluate_downstream(batch_idx, embeddings.device)
        return loss

    def evaluate_downstream(self, batch_idx, device):
        if self.args.ds_every_n_epoch > 0 and self.global_rank == 0 and batch_idx == 0 and (
                self.current_epoch + 1) % self.args.ds_every_n_epoch == 0:
            extract_embedding('train', self.ds_loader_train, self.model, self.args.output_dir, device, self.object_box)
            extract_embedding('val', self.ds_loader_val, self.model, self.args.output_dir, device, self.object_box)

            if '1' in self.args.eval_task:  # classification
                regular_f1, ego2exo_val_f1, exo2ego_val_f1 = classification(self.args.output_dir,
                                                                            self.ds_dataset_train.video_ego_id,
                                                                            self.ds_dataset_val.video_ego_id)
                self.log('regular_f1', regular_f1)
                self.log('ego2exo_val_f1', ego2exo_val_f1)
                self.log('exo2ego_val_f1', exo2ego_val_f1)

            if '2' in self.args.eval_task:  # retrieval
                regular_map10, ego2exo_val_map10, exo2ego_val_map10 = frame_retrieval(self.args.output_dir,
                                                                                      self.ds_dataset_val.video_len_list,
                                                                                      self.ds_dataset_val.video_paths1)
                self.log('regular_map10', float(regular_map10))
                self.log('ego2exo_val_map10', float(ego2exo_val_map10))
                self.log('exo2ego_val_map10', float(exo2ego_val_map10))

            if '3' in self.args.eval_task:  # event completion
                modify_embeddings = True if self.args.dataset == 'pour_liquid' else False  # augment embedding for pour_liquid
                train_score, val_score = compute_progression_value(self.args.output_dir, self.ds_dataset_train.video_len_list,
                                                                   self.ds_dataset_val.video_len_list, modify_embeddings)
                self.log('train_score', train_score)
                self.log('val_score', val_score)

            if '4' in self.args.eval_task:  # kendall's tau
                train_tau = kendalls_tau(self.args.output_dir, self.ds_dataset_train.video_len_list,
                                         self.ds_dataset_train.video_paths1, 'train', False)
                val_tau = kendalls_tau(self.args.output_dir, self.ds_dataset_val.video_len_list,
                                       self.ds_dataset_val.video_paths1, 'val', False)
                self.log('train_tau', train_tau)
                self.log('val_tau', val_tau)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        dataset = VideoAlignmentTrainDataset(self.args, 'train')
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers)
        return loader

    def val_dataloader(self):
        dataset = VideoAlignmentTrainDataset(self.args, 'val')
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers)
        return loader


class ObjectVideoAlignment(VideoAlignment):
    def __init__(self, args):
        super(ObjectVideoAlignment, self).__init__(args)
        self.model = RoIPosEmbedder(args)

    def model_forward(self, x, bbox):
        embeds_list = []
        for i in range(2):
            embeds = self.model(x[:, i, ...].permute(0, 1, 4, 2, 3), bbox[:, i, :])
            embeds_list.append(embeds)
        embeddings = torch.stack((embeds_list[0], embeds_list[1]), dim=1)
        return embeddings

    def training_step(self, batch, batch_idx):
        frames, steps, seq_lens, bbox, pos_steps = batch
        embeddings = self.model_forward(frames, bbox)
        loss, loss2 = self.loss(embeddings, steps, seq_lens, pos_steps, self.global_step)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_loss_reg', loss2, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, steps, seq_lens, bbox, pos_steps = batch
        embeddings = self.model_forward(frames, bbox)
        loss, loss2 = self.loss(embeddings, steps, seq_lens, pos_steps, self.global_step)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_loss_reg', loss2, on_step=True, on_epoch=True)
        self.evaluate_downstream(batch_idx, embeddings.device)
        return loss

    def train_dataloader(self):
        dataset = VideoAlignmentBboxTrainDataset(self.args, 'train')
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers)
        return loader

    def val_dataloader(self):
        dataset = VideoAlignmentBboxTrainDataset(self.args, 'val')
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers)
        return loader

