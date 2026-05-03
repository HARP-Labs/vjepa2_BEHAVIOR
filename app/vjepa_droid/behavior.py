# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu

from app.vjepa_droid.droid import DROIDVideoDataset

logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=None,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
):
    dataset = BehaviorVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("Behavior video data loader created")
    return data_loader, dist_sampler


class BehaviorVideoDataset(DROIDVideoDataset):
    """BEHAVIOR dataset that reuses DROID preprocessing and action/state generation."""

    def __init__(
        self,
        data_path,
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
    ):
        self.data_path = data_path
        self.dataset_root = os.path.dirname(os.path.abspath(data_path))
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        manifest = self._load_manifest(data_path)
        self.samples = self._parse_samples(manifest)

    def _load_manifest(self, manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _resolve_episode_layout(self, task_name, episode_file):
        if task_name is not None and episode_file is not None:
            episode_name = os.path.splitext(os.path.basename(episode_file))[0]
            base = os.path.join(self.dataset_root, task_name)
            return {
                "video": os.path.join(base, "video", f"{episode_name}.mp4"),
                "parquet": os.path.join(base, "data", f"{episode_name}.parquet"),
            }
        return None

    def _parse_samples(self, manifest):
        samples = []
        for ep in manifest.get("episodes", []):
            task_name = ep.get("task_name")
            episode_file = ep.get("episode_file")
            fallback = self._resolve_episode_layout(task_name, episode_file)

            video_rel = ep.get("video_file") or (ep.get("video_files") or [None])[0]
            parquet_rel = ep.get("data_parquet_file")

            video_path = (
                os.path.join(self.dataset_root, video_rel)
                if video_rel is not None
                else (fallback["video"] if fallback is not None else None)
            )
            parquet_path = (
                os.path.join(self.dataset_root, parquet_rel)
                if parquet_rel is not None
                else (fallback["parquet"] if fallback is not None else None)
            )

            if video_path is None or parquet_path is None:
                continue

            samples.append({"video_path": video_path, "parquet_path": parquet_path})

        if not samples:
            raise ValueError(f"No episodes found in manifest: {self.data_path}")
        return samples

    def loadvideo_decord(self, sample):
        vpath = sample["video_path"]
        ppath = sample["parquet_path"]

        df = pd.read_parquet(ppath)
        pose_cols = [
            c
            for c in [
                "observation.state.pose.x",
                "observation.state.pose.y",
                "observation.state.pose.z",
                "observation.state.pose.roll",
                "observation.state.pose.pitch",
                "observation.state.pose.yaw",
                "observation.state.gripper",
            ]
            if c in df.columns
        ]
        if len(pose_cols) < 7:
            raise ValueError(f"Missing state columns in parquet: {ppath}")

        states = df[pose_cols].to_numpy(dtype=np.float32)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))

        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes or states.shape[0] < nframes:
            raise Exception(f"Episode too short {vpath=}, {nframes=}, {vlen=}, states={states.shape[0]}")

        ef = np.random.randint(nframes, min(vlen, states.shape[0]))
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

        states = states[indices, :][:: self.frameskip]
        actions = self.poses_to_diffs(states)

        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        # No extrinsics in BEHAVIOR parquet for now; keep predictor API-compatible.
        extrinsics = np.zeros((states.shape[0], 6), dtype=np.float32)
        return buffer, actions, states, extrinsics, indices

    def __getitem__(self, index):
        sample = self.samples[index]

        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(sample)
                loaded_video = True
            except Exception as e:
                logger.info(f"Encountered exception when loading sample={sample} {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        return buffer, actions, states, extrinsics, indices
