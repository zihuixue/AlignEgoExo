import os
import pickle
import random
import numpy as np

from dataset.video_align_dataset import VideoAlignmentTrainDataset, VideoAlignmentDownstreamDataset
from utils.util import _extract_frames_h5py, get_num_frames


def expand_bbox(bbox_array, expansion_ratio):
    # input [batch_size, num_objects, 10]

    # Compute the width and height of each bounding box
    widths = bbox_array[:, :, 2] - bbox_array[:, :, 0]
    heights = bbox_array[:, :, 3] - bbox_array[:, :, 1]

    # Expand each bounding box by the given expansion ratio
    new_widths = expansion_ratio * widths
    new_heights = expansion_ratio * heights

    # Compute the x and y offsets for each bounding box
    x_offsets = 0.5 * (new_widths - widths)
    y_offsets = 0.5 * (new_heights - heights)

    # Create a new array with the expanded bounding boxes
    new_bbox_array = bbox_array.copy()
    new_bbox_array[:, :, 0] -= x_offsets
    new_bbox_array[:, :, 1] -= y_offsets
    new_bbox_array[:, :, 2] += x_offsets
    new_bbox_array[:, :, 3] += y_offsets

    # Reshape the output array to [N, 12] for consistency
    return new_bbox_array


class VideoAlignmentBboxTrainDataset(VideoAlignmentTrainDataset):
    def __init__(self, args, mode):
        super(VideoAlignmentBboxTrainDataset, self).__init__(args, mode)
        self.sample_by_bbox = args.sample_by_bbox
        self.bbox_threshold = args.bbox_threshold
        self._load_bounding_box()
        if args.dataset != 'tennis_forehand':
            sx = self.args.input_size / self.video_res[0]
            sy = self.args.input_size / self.video_res[1]
            self.scale_factor = np.array([sx, sy, sx, sy])
            self.normalize_constants = np.array([self.video_res[0], self.video_res[1], self.video_res[0], self.video_res[1]])
        else:
            sx_ego, sy_ego = self.args.input_size / self.video_res_ego[0], self.args.input_size / self.video_res_ego[1]
            sx_exo, sy_exo = self.args.input_size / self.video_res_exo[0], self.args.input_size / self.video_res_exo[1]
            self.scale_factor_ego = np.array([sx_ego, sy_ego, sx_ego, sy_ego])
            self.scale_factor_exo = np.array([sx_exo, sy_exo, sx_exo, sy_exo])

        self.expansion_ratio = args.bbox_expansion_ratio
        self.shuffle_num = args.dtw_shuffle_num

    def _load_bounding_box(self):
        with open(os.path.join(self.data_path, 'det_bounding_box.pickle'), 'rb') as handle:
            self.bounding_box_dict = pickle.load(handle)
        if self.bbox_threshold > 0.0:
            for key, value in self.bounding_box_dict.items():
                mask = value[:, :, 4] < self.bbox_threshold
                replacement = np.zeros_like(value)
                replacement[..., -1] = -1
                value[mask] = replacement[mask]
                self.bounding_box_dict[key] = value

    def __getitem__(self, idx):
        selected_videos = [random.sample(self.video_paths1, 1), random.sample(self.video_paths2, 1)]
        final_frames = list()
        seq_lens = list()
        steps = list()
        bbs = list()
        pos_list = list()
        for video in selected_videos:
            video = video[0]
            video_name = video.replace('ego/', 'ego_').replace('exo/', 'exo_').replace('.mp4', '').split('/')[-1]
            bounding_box = self.bounding_box_dict[video_name].copy()
            if self.dataset == 'tennis_forehand':
                if 'exo' in video:
                    bounding_box[:, :, 0:4] = bounding_box[:, :, 0:4] * self.scale_factor_exo
                else:
                    bounding_box[:, :, 0:4] = bounding_box[:, :, 0:4] * self.scale_factor_ego
            else:
                bounding_box[:, :, 0:4] = bounding_box[:, :, 0:4] * self.scale_factor
            bounding_box = expand_bbox(bounding_box, self.expansion_ratio)
            if self.args.one_object_bbox:
                bounding_box[:, -1, -1] = -1  # set last object detection result to be null
            video_frames_count = bounding_box.shape[0]
            video_frames_count_true = get_num_frames(video)
            if video_frames_count != video_frames_count_true:
                # print(f'problematic video', video, video_frames_count, video_frames_count_true)
                bounding_box = bounding_box[0:video_frames_count_true]
                video_frames_count = bounding_box.shape[0]
            assert video_frames_count == get_num_frames(video)

            if self.sample_by_bbox: # sample pos indices by bbox prob
                main_frames, selected_frames = self.sample_frames(video_frames_count)
                bbox_prob = bounding_box[:, :, 4].mean(axis=-1)
                prob = bbox_prob[main_frames]
                prob = prob / prob.sum()
                pos_indices = np.random.choice(range(self.num_steps), size=self.shuffle_num, replace=False, p=prob)
                pos_indices = np.sort(pos_indices)

            else:
                main_frames, selected_frames = self.sample_frames(video_frames_count)
                segment_size = self.num_steps // self.shuffle_num
                pos_indices = [int((i + 0.5) * segment_size) for i in range(self.shuffle_num)]

            h5_file_name = _extract_frames_h5py(video, self.frame_save_path)

            frames = self.get_frames_h5py(h5_file_name, selected_frames)
            frames = np.array(frames)  # (64, 168, 168, 3)

            final_frames.append(np.expand_dims(frames.astype(np.float32), axis=0))
            steps.append(np.expand_dims(np.array(main_frames), axis=0))
            seq_lens.append(video_frames_count)
            bbs.append(np.expand_dims(bounding_box[main_frames].astype(np.float32), axis=0))
            pos_list.append(pos_indices)

        return (
            np.concatenate(final_frames),
            np.concatenate(steps),
            np.array(seq_lens),
            np.concatenate(bbs),  # (2, 32, 4, 10), 4-object+hand num, 10-attribute num
            np.array(pos_list)  # (2, 4)
        )


class VideoAlignmentBboxDownstreamDataset(VideoAlignmentDownstreamDataset):
    def __init__(self, args, mode):
        super(VideoAlignmentBboxDownstreamDataset, self).__init__(args, mode)
        self.bbox_threshold = args.bbox_threshold
        self._load_bounding_box()
        if args.dataset != 'tennis_forehand':
            sx = self.args.input_size / self.video_res[0]
            sy = self.args.input_size / self.video_res[1]
            self.scale_factor = np.array([sx, sy, sx, sy])
        else:
            sx_ego, sy_ego = self.args.input_size / self.video_res_ego[0], self.args.input_size / \
                             self.video_res_ego[1]
            sx_exo, sy_exo = self.args.input_size / self.video_res_exo[0], self.args.input_size / \
                             self.video_res_exo[1]
            self.scale_factor_ego = np.array([sx_ego, sy_ego, sx_ego, sy_ego])
            self.scale_factor_exo = np.array([sx_exo, sy_exo, sx_exo, sy_exo])
        self.expansion_ratio = args.bbox_expansion_ratio

    def _load_bounding_box(self):
        with open(os.path.join(self.data_path, 'det_bounding_box.pickle'), 'rb') as handle:
            self.bounding_box_dict = pickle.load(handle)
        if self.bbox_threshold > 0.0:
            for key, value in self.bounding_box_dict.items():
                mask = value[:, :, 4] < self.bbox_threshold
                replacement = np.zeros_like(value)
                replacement[..., -1] = -1
                value[mask] = replacement[mask]
                self.bounding_box_dict[key] = value

    def __getitem__(self, idx):
        video_path, frame_id, frame_label = self.frame_path_list[idx]
        h5_file_name = _extract_frames_h5py(video_path, self.frame_save_path)
        context_frame_id = max(0, frame_id - self.frame_stride)
        frame = self.get_frames_h5py(h5_file_name, [context_frame_id, frame_id])
        frame = np.array(frame).astype(np.float32)  # (2, 168, 168, 3)

        video_name = video_path.replace('ego/', 'ego_').replace('exo/', 'exo_').replace('.mp4', '').split('/')[-1]
        bounding_box = self.bounding_box_dict[video_name].copy()
        if self.dataset == 'tennis_forehand':
            if 'exo' in video_name:
                bounding_box[:, :, 0:4] = bounding_box[:, :, 0:4] * self.scale_factor_exo
            else:
                bounding_box[:, :, 0:4] = bounding_box[:, :, 0:4] * self.scale_factor_ego
        else:
            bounding_box[:, :, 0:4] = bounding_box[:, :, 0:4] * self.scale_factor
        bounding_box = expand_bbox(bounding_box, self.expansion_ratio)
        if self.args.one_object_bbox:
            bounding_box[:, -1, -1] = -1  # set last object detection result to be null
        bounding_box = bounding_box[frame_id]

        return frame, frame_label, video_path, bounding_box.astype(np.float32)
