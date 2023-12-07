import os
import random

import cv2
import h5py
import pickle
import numpy as np
import torch

from torch.utils.data import Dataset
from utils.util import _extract_frames_h5py, get_num_frames


data_video_resolution = {
    'break_eggs': [1024, 768],
    'pour_milk': [640, 360],
    'pour_liquid': [320, 240],
    'tennis_ego': [1920, 1080],
    'tennis_exo': [480, 360]
}


class VideoAlignmentDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.dataset = args.dataset
        assert args.dataset in ['break_eggs', 'pour_milk', 'pour_liquid', 'tennis_forehand']
        self.data_path = os.path.join(args.dataset_root, args.dataset)
        if args.dataset != 'tennis_forehand':
            self.video_res = data_video_resolution[args.dataset]
        else:
            self.video_res_ego = data_video_resolution['tennis_ego']
            self.video_res_exo = data_video_resolution['tennis_exo']

        self.num_steps = args.num_frames
        self.frame_stride = args.frame_stride
        self.random_offset = args.random_offset

        if args.dataset in ['break_eggs', 'tennis_forehand']:
            self.video_paths1 = self._construct_video_path_by_mode(os.path.join(self.data_path, args.view1), mode)
            self.video_paths2 = self._construct_video_path_by_mode(os.path.join(self.data_path, args.view2), mode)
            self.frame_save_path = self.data_path
        else:
            self.video_paths1 = self._construct_video_path(os.path.join(self.data_path, mode, args.view1))
            self.video_paths2 = self._construct_video_path(os.path.join(self.data_path, mode, args.view2))
            self.frame_save_path = os.path.join(self.data_path, mode)

        self.merge_all = True    # setting merge_all to False -> separate ego-exo training
        if self.merge_all:
            tmp_path = list(set(self.video_paths1 + self.video_paths2))
            self.video_paths1 = tmp_path
            self.video_paths2 = tmp_path

        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _construct_video_path(self, dir_name):
        video_paths = []
        for item in os.listdir(dir_name):
            if item.endswith('.mp4'):
                video_paths.append(os.path.join(dir_name, item))
        assert len(video_paths) > 1
        print(f'{len(video_paths)} videos in {dir_name}')
        return video_paths

    def _construct_video_path_by_mode(self, dir_name, mode):
        video_paths = []
        f_out = open(os.path.join(dir_name, mode+'.csv'), 'r')
        for line in f_out.readlines():
            line = line.strip()
            video_paths.append(os.path.join(dir_name, line))
        return video_paths

    def get_frames_h5py(self, h5_file_path, frames_list, bbox_list=None):
        final_frames = list()
        h5_file = h5py.File(h5_file_path, 'r')
        frames = h5_file['images']
        for frame_num in frames_list:
            frame_ = frames[frame_num]
            frame = cv2.resize(
                frame_,
                (self.args.input_size, self.args.input_size),
                interpolation=cv2.INTER_AREA
            )
            frame = (frame / 127.5) - 1.0

            final_frames.append(frame)

        h5_file.close()
        assert len(final_frames) == len(frames_list)
        return final_frames

    def get_steps(self, step):
        """Sample multiple context steps for a given step."""
        if self.num_steps < 1:
            raise ValueError('num_steps should be >= 1.')
        if self.frame_stride < 1:
            raise ValueError('stride should be >= 1.')
        steps = torch.arange(step - self.frame_stride, step + self.frame_stride, self.frame_stride)
        return steps

    def sample_frames(self, seq_len):
        # sampling_strategy = 'offset_uniform' for now
        assert seq_len >= self.random_offset
        if self.num_steps < seq_len - self.random_offset:
            # random sample
            steps = torch.randperm(seq_len - self.random_offset) + self.random_offset
            steps = steps[:self.num_steps]
            steps = torch.sort(steps)[0]
        else:
            # sample all
            steps = torch.arange(0, self.num_steps, dtype=torch.int64)
        chosen_steps = torch.clamp(steps, 0, seq_len - 1)
        steps = torch.cat(list(map(self.get_steps, steps)), dim=-1)
        steps = torch.clamp(steps, 0, seq_len - 1)
        return chosen_steps, steps


class VideoAlignmentTrainDataset(VideoAlignmentDataset):
    def __init__(self, args, mode):
        super(VideoAlignmentTrainDataset, self).__init__(args, mode)

    def __len__(self):
        return len(self.video_paths1)

    def __getitem__(self, idx):
        selected_videos = [random.sample(self.video_paths1, 1), random.sample(self.video_paths2, 1)]
        final_frames = list()
        seq_lens = list()
        steps = list()
        for video in selected_videos:
            video = video[0]
            video_frames_count = get_num_frames(video)
            main_frames, selected_frames = self.sample_frames(video_frames_count)
            h5_file_name = _extract_frames_h5py(
                video,
                self.frame_save_path
            )
            frames = self.get_frames_h5py(
                h5_file_name,
                selected_frames,
            )
            frames = np.array(frames)  # (64, 168, 168, 3)
            final_frames.append(
                np.expand_dims(frames.astype(np.float32), axis=0)
            )
            steps.append(np.expand_dims(np.array(main_frames), axis=0))
            seq_lens.append(video_frames_count)

        return (
            np.concatenate(final_frames),
            np.concatenate(steps),
            np.array(seq_lens)
        )


class VideoAlignmentDownstreamDataset(VideoAlignmentDataset):
    def __init__(self, args, mode):
        args.merge_all = True
        super(VideoAlignmentDownstreamDataset, self).__init__(args, mode)
        self.video_paths1 = sorted(self.video_paths1)
        self._load_label()
        self._construct_frame_path()

    def _construct_frame_path(self):
        self.frame_path_list = []
        self.video_len_list = []
        self.video_ego_id = []
        for video in self.video_paths1:
            video_frames_count = get_num_frames(video)
            self.video_len_list.append(video_frames_count)
            video_name = video.replace('.mp4', '').split('/')[-1]
            view = video.split('/')[-2]
            labels = self.label_dict[video_name]
            assert video_frames_count == len(labels)
            for frame_id in range(video_frames_count):
                self.frame_path_list.append([video, frame_id, labels[frame_id]])
                if view == 'ego':
                    self.video_ego_id.append(1)
                else:
                    self.video_ego_id.append(0)
        print(f'Finish constructing frames path list, total len {len(self.frame_path_list)}')

    def _load_label(self):
        file_path = os.path.join(self.data_path, 'label.pickle')
        with open(file_path, 'rb') as handle:
            self.label_dict = pickle.load(handle)

    def __len__(self):
        return len(self.frame_path_list)

    def __getitem__(self, idx):
        video_path, frame_id, frame_label = self.frame_path_list[idx]
        h5_file_name = _extract_frames_h5py(video_path, self.frame_save_path)
        context_frame_id = max(0, frame_id - self.frame_stride)
        frame = self.get_frames_h5py(h5_file_name, [context_frame_id, frame_id])
        frame = np.array(frame).astype(np.float32)  # (2, 168, 168, 3)
        return frame, frame_label, video_path