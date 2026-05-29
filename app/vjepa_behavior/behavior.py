# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import getLogger

import numpy as np
import torch
import torch.utils.data
from streaming import StreamingDataset

logger = getLogger()

# Column names in the MDS schema, in the order they will be concatenated.
_CAMERA_KEYS = {
    "head": "tokens_head",
    "left_wrist": "tokens_left_wrist",
    "right_wrist": "tokens_right_wrist",
}


class BehaviorMDSDataset(torch.utils.data.Dataset):
    """
    Loads consecutive T-frame clips from the BEHAVIOR-1K pre-encoded MDS dataset.

    Each MDS row is one encoded video frame (one step). Rows belonging to the
    same episode are contiguous and sorted by step_pos. The dataset builds a
    clip index at init time (scanning episode_idx / step_pos / episode_len only,
    no token loading) and caches it to disk for fast subsequent runs.

    Returns per sample:
        tokens  (T, N_cams * tpf_per_cam, embed_dim)  float16
        actions (T, action_fstp * 23)                  float32
        states  (T, 133)                               float32
    """

    def __init__(
        self,
        remote,
        local,
        cameras,
        frames_per_clip,
        shuffle=False,
    ):
        self.cameras = cameras
        self.frames_per_clip = frames_per_clip

        self._ds = StreamingDataset(remote=remote, local=local, shuffle=shuffle)

        index_path = os.path.join(local, f"_clip_index_T{frames_per_clip}.npy")
        if os.path.exists(index_path):
            logger.info(f"Loading clip index from {index_path}")
            self.clips_index = np.load(index_path)
        else:
            logger.info("Building clip index (scanning episode metadata)...")
            self.clips_index = self._build_index()
            os.makedirs(local, exist_ok=True)
            np.save(index_path, self.clips_index)
            logger.info(f"Clip index saved to {index_path} ({len(self.clips_index)} clips)")

    def _build_index(self):
        """
        Scan all rows reading only episode_idx / step_pos / episode_len.
        Returns row indices of valid clip starts (at least frames_per_clip steps remain).
        """
        T = self.frames_per_clip
        valid_starts = []
        n = len(self._ds)
        for i in range(n):
            row = self._ds[i]
            step_pos = int(row["step_pos"])
            episode_len = int(row["episode_len"])
            if episode_len - step_pos >= T:
                valid_starts.append(i)
        return np.array(valid_starts, dtype=np.int64)

    def __len__(self):
        return len(self.clips_index)

    def __getitem__(self, idx):
        row_start = int(self.clips_index[idx])
        T = self.frames_per_clip

        rows = [self._ds[row_start + t] for t in range(T)]

        # --- tokens: concatenate active camera views along token dim ---
        cam_token_list = []
        for cam in self.cameras:
            key = _CAMERA_KEYS[cam]
            # shape per row: (tpf_per_cam, embed_dim) float16
            cam_tokens = np.stack([row[key] for row in rows], axis=0)  # (T, tpf, D)
            cam_token_list.append(cam_tokens)
        # (T, N_cams * tpf_per_cam, embed_dim)
        tokens = np.concatenate(cam_token_list, axis=1).astype(np.float16)

        # --- actions: (T, action_fstp * 23) ---
        actions = np.stack([row["actions"].astype(np.float32) for row in rows], axis=0)

        # --- states: (T, 133) ---
        states = np.stack([row["states"].astype(np.float32) for row in rows], axis=0)

        return (
            torch.from_numpy(tokens),
            torch.from_numpy(actions),
            torch.from_numpy(states),
        )


def make_behavior_dataset(
    remote,
    local,
    cameras,
    frames_per_clip,
    batch_size,
    rank=0,
    world_size=1,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    drop_last=True,
    collator=None,
):
    dataset = BehaviorMDSDataset(
        remote=remote,
        local=local,
        cameras=cameras,
        frames_per_clip=frames_per_clip,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator if collator is not None else torch.utils.data.default_collate,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info(f"BehaviorMDSDataset loader created: {len(dataset)} clips, {len(data_loader)} batches/rank")
    return dataset, data_loader, dist_sampler
