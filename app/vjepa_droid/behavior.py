import io
import json
import os
from logging import getLogger
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from huggingface_hub import HfApi

logger = getLogger()


class BehaviorVideoDataset(torch.utils.data.Dataset):
    """BEHAVIOR dataset with deterministic episode-chunk sampling for pre-encoding/training."""

    def __init__(
        self,
        data_path,
        fpcs=16,
        fps=5,
        transform=None,
        camera_frame=False,
        state_start_idx=0,
        state_dim=7,
        action_dim=23,
    ):
        self.data_path = data_path
        self.dataset_root = os.path.dirname(os.path.abspath(data_path))
        self.fpc = fpcs
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.state_start_idx = state_start_idx
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._parquet_cache = {}
        self._video_reader_cache = {}

        manifest = self._load_manifest(data_path)
        self.samples = self._parse_samples(manifest)
        self.episode_plans = self._build_episode_plans()
        self.windows = self._build_window_index()

    def _load_manifest(self, manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _parse_samples(self, manifest):
        samples = []
        for ep in manifest.get("episodes", []):
            task_name = ep.get("task_name")
            episode_file = ep.get("episode_file")
            if task_name is None or episode_file is None:
                logger.warning(f"Skipping manifest entry missing task_name/episode_file: {ep}")
                continue
            episode_name = os.path.splitext(os.path.basename(episode_file))[0]
            base = os.path.join(self.dataset_root, task_name)
            video_path = os.path.join(base, "video", f"{episode_name}.mp4")
            parquet_path = os.path.join(base, "data", f"{episode_name}.parquet")
            samples.append({
                "video_path": video_path,
                "parquet_path": parquet_path,
            })

        if not samples:
            raise ValueError(f"No episodes found in manifest: {self.data_path}")

        return samples

    def _build_episode_plans(self):
        plans = []
        for sample_idx, sample in enumerate(self.samples):
            try:
                indices, fstp, max_len = self._episode_sampled_indices(sample)
                if indices is None or len(indices) == 0:
                    logger.warning(f"Skipping sample due to insufficient frames: {sample}")
                    continue
                plans.append(
                    {
                        "sample_idx": sample_idx,
                        "indices": indices,
                        "fstp": fstp,
                        "max_len": max_len,
                    }
                )
            except Exception as e:
                logger.warning(f"Skipping sample during episode planning sample={sample} {e=}")
        if not plans:
            raise ValueError(f"No valid episode plans found in manifest: {self.data_path}")
        logger.info(f"Built {len(plans)} valid episode plans")
        return plans

    def _episode_sampled_indices(self, sample):
        vpath = sample["video_path"]
        ppath = sample["parquet_path"]
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        vfps = vr.get_avg_fps()
        fps = self.fps if self.fps is not None else vfps
        if fps <= 0:
            raise ValueError(f"fps must be > 0. Got fps={fps} for {vpath}")
        fstp = max(1, ceil(vfps / fps))
        vlen = len(vr)
        parquet_len = len(pd.read_parquet(ppath, columns=["action"]))
        if abs(vlen - parquet_len) > 2:
            logger.warning(f"Length mismatch {vpath}: video={vlen}, parquet={parquet_len}")
        max_len = min(vlen, parquet_len)
        if max_len < fstp:
            logger.warning(f"Too short episode {vpath}: max_len={max_len}, fstp={fstp}")
            return None, fstp, max_len
        indices = np.arange(0, max_len, fstp, dtype=np.int64)
        return indices, fstp, max_len

    def _build_window_index(self):
        windows = []
        for episode_idx, plan in enumerate(self.episode_plans):
            indices = plan["indices"]
            n = len(indices)
            if n == 0:
                logger.warning(f"Skipping episode_idx={episode_idx} with no sampled indices")
                continue
            for start in range(0, n, self.fpc):
                windows.append((episode_idx, start))
        if not windows:
            raise ValueError(f"No valid windows found in manifest: {self.data_path}")
        logger.info(
            f"Built BEHAVIOR window index with {len(windows)} non-overlapping windows "
            f"from {len(self.episode_plans)} valid episode plans"
        )
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_idx, start_idx = self.windows[index]
        plan = self.episode_plans[episode_idx]
        sample = self.samples[plan["sample_idx"]]
        while True:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(sample, plan, start_idx=start_idx)
                break
            except Exception as e:
                logger.warning(f"Encountered exception when loading sample={sample} {e=}")
                episode_idx, start_idx = self.windows[np.random.randint(self.__len__())]
                plan = self.episode_plans[episode_idx]
                sample = self.samples[plan["sample_idx"]]

        valid_len = min(self.fpc, len(plan["indices"]) - start_idx)
        return {
            "video": buffer,
            "actions": actions,
            "states": states,
            "extrinsics": extrinsics,
            "frame_indices": indices,
            "episode_idx": episode_idx,
            "start_idx": start_idx,
            "valid_len": valid_len,
        }

    def loadvideo_decord(self, sample, plan, start_idx=0):
        vpath = sample["video_path"]
        ppath = sample["parquet_path"]
        df = self._load_parquet(ppath)
        if "observation.state" not in df.columns or "action" not in df.columns:
            raise ValueError(f"Expected `observation.state` and `action` in parquet: {ppath}")
        full_states = np.asarray(df["observation.state"].to_list(), dtype=np.float32)
        full_actions = np.asarray(df["action"].to_list(), dtype=np.float32)

        if full_actions.shape[1] < self.action_dim:
            raise ValueError(f"Action dim out of bounds for {ppath}: {full_actions.shape[1]=}, {self.action_dim=}")
        if full_states.shape[1] < self.state_start_idx + self.state_dim:
            raise ValueError(
                f"State slice out of bounds for {ppath}: {full_states.shape[1]=}, {self.state_start_idx=}, {self.state_dim=}"
            )

        states = full_states[:, self.state_start_idx : self.state_start_idx + self.state_dim]
        vr = self._get_video_reader(vpath)
        fstp = plan["fstp"]
        max_len = min(plan["max_len"], states.shape[0], full_actions.shape[0], len(vr))
        indices = plan["indices"]

        if len(indices) == 0:
            raise RuntimeError(f"No indices in episode plan for {vpath=}, {fstp=}, {max_len=}")

        end_idx = min(start_idx + self.fpc, len(indices))
        window_indices = indices[start_idx:end_idx]
        if len(window_indices) < self.fpc:
            pad = np.full((self.fpc - len(window_indices),), window_indices[-1], dtype=np.int64)
            window_indices = np.concatenate([window_indices, pad])

        raw_states = states
        raw_actions = full_actions[:, : self.action_dim]
        states = []
        actions = []
        for i, start in enumerate(window_indices):
            end = window_indices[i + 1] if i + 1 < len(window_indices) else min(start + fstp, max_len)
            state_chunk = raw_states[start:end]
            action_chunk = raw_actions[start:end]

            if len(state_chunk) == 0:
                logger.warning(f"Empty state chunk for {vpath=}, {start=}, {end=}")
                state_chunk = np.zeros((fstp, self.state_dim), dtype=np.float32)
            elif len(state_chunk) < fstp:
                pad = np.repeat(state_chunk[-1:], fstp - len(state_chunk), axis=0)
                state_chunk = np.concatenate([state_chunk, pad], axis=0)
            else:
                state_chunk = state_chunk[:fstp]

            if len(action_chunk) == 0:
                logger.warning(f"Empty action chunk for {vpath=}, {start=}, {end=}")
                action_chunk = np.zeros((fstp, self.action_dim), dtype=np.float32)
            elif len(action_chunk) < fstp:
                pad = np.repeat(action_chunk[-1:], fstp - len(action_chunk), axis=0)
                action_chunk = np.concatenate([action_chunk, pad], axis=0)
            else:
                action_chunk = action_chunk[:fstp]

            states.append(state_chunk.reshape(fstp * self.state_dim))
            actions.append(action_chunk.reshape(fstp * self.action_dim))

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        vr.seek(0)
        buffer = vr.get_batch(window_indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)
        extrinsics = np.zeros((states.shape[0], 6), dtype=np.float32)
        return buffer, actions, states, extrinsics, window_indices

    def _load_parquet(self, ppath):
        cached = self._parquet_cache.get(ppath)
        if cached is not None:
            return cached
        df = pd.read_parquet(ppath)
        self._parquet_cache[ppath] = df
        return df

    def _get_video_reader(self, vpath):
        cached = self._video_reader_cache.get(vpath)
        if cached is not None:
            return cached
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        self._video_reader_cache[vpath] = vr
        return vr


class BehaviorEpisodePreencoder:
    """Run a vision encoder on BEHAVIOR clips and save pre-encoded episode shards."""

    def __init__(self, encoder, device=None, dtype=torch.float32):
        self.encoder = encoder.eval()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        self.encoder.to(self.device)

    def _to_video_tensor(self, video):
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got shape={tuple(video.shape)}")
        if video.shape[-1] in (1, 3):
            video = video.permute(0, 4, 1, 2, 3)
        return video.to(self.device, dtype=self.dtype, non_blocking=True)

    @staticmethod
    def behavior_preencode_collate(batch):
        return {
            "video": np.stack([item["video"] for item in batch], axis=0),
            "actions": np.stack([item["actions"] for item in batch], axis=0),
            "states": np.stack([item["states"] for item in batch], axis=0),
            "frame_indices": np.stack([item["frame_indices"] for item in batch], axis=0),
            "episode_idx": np.asarray([item["episode_idx"] for item in batch], dtype=np.int64),
            "start_idx": np.asarray([item["start_idx"] for item in batch], dtype=np.int64),
            "valid_len": np.asarray([item["valid_len"] for item in batch], dtype=np.int64),
        }

    @torch.no_grad()
    def encode_full_episodes(self, dataset, output_dir=None, hf_repo_id=None, hf_path_prefix="", episodes_per_shard=1, batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if not output_dir and not hf_repo_id:
            raise ValueError("Either output_dir or hf_repo_id must be provided")

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=self.behavior_preencode_collate,
        )
        episode_buffers = {episode_idx: {"tokens": [], "actions": [], "states": [], "frame_indices": [], "starts": []} for episode_idx in range(len(dataset.episode_plans))}

        for batch in data_loader:
            tokens = self.encoder(self._to_video_tensor(batch["video"]))
            if isinstance(tokens, (tuple, list)):
                tokens = tokens[0]
            tokens = tokens.detach().cpu().float().numpy()
            for b in range(tokens.shape[0]):
                episode_idx = int(batch["episode_idx"][b])
                valid_len = int(batch["valid_len"][b])
                episode_buffers[episode_idx]["tokens"].append(tokens[b, :valid_len])
                episode_buffers[episode_idx]["actions"].append(batch["actions"][b, :valid_len])
                episode_buffers[episode_idx]["states"].append(batch["states"][b, :valid_len])
                episode_buffers[episode_idx]["frame_indices"].append(batch["frame_indices"][b, :valid_len])
                episode_buffers[episode_idx]["starts"].append(int(batch["start_idx"][b]))

        shard_data, shard_id, encoded_episodes = [], 0, 0
        for episode_idx, buffer in episode_buffers.items():
            if not buffer["tokens"]:
                logger.warning(f"Skipping empty encoded episode {episode_idx}")
                continue
            order = np.argsort(buffer["starts"])
            shard_data.append({
                "episode_idx": int(episode_idx),
                "sample_idx": int(dataset.episode_plans[episode_idx]["sample_idx"]),
                "tokens": np.concatenate([buffer["tokens"][i] for i in order], axis=0),
                "actions": np.concatenate([buffer["actions"][i] for i in order], axis=0),
                "states": np.concatenate([buffer["states"][i] for i in order], axis=0),
                "frame_indices": np.concatenate([buffer["frame_indices"][i] for i in order], axis=0),
            })
            encoded_episodes += 1
            if len(shard_data) >= episodes_per_shard:
                self._write_shard(output_dir=output_dir, hf_repo_id=hf_repo_id, hf_path_prefix=hf_path_prefix, shard_id=shard_id, shard_data=shard_data)
                shard_data, shard_id = [], shard_id + 1

        if shard_data:
            self._write_shard(output_dir=output_dir, hf_repo_id=hf_repo_id, hf_path_prefix=hf_path_prefix, shard_id=shard_id, shard_data=shard_data)

        logger.info(f"Pre-encoding finished: {encoded_episodes} episodes encoded")

    def _write_shard(self, shard_id, shard_data, output_dir=None, hf_repo_id=None, hf_path_prefix="", hf_repo_type="dataset"):
        shard_name = f"behavior_preencoded_shard_{shard_id:05d}.pt"
        if output_dir:
            save_path = os.path.join(output_dir, shard_name)
            torch.save(shard_data, save_path)
            logger.info(f"Saved shard {shard_id} with {len(shard_data)} items at {save_path}")
        if hf_repo_id:
            bio = io.BytesIO()
            torch.save(shard_data, bio)
            bio.seek(0)
            path_in_repo = f"{hf_path_prefix.strip('/')}/{shard_name}".lstrip("/")
            HfApi().upload_file(path_or_fileobj=bio, path_in_repo=path_in_repo, repo_id=hf_repo_id, repo_type=hf_repo_type)
            logger.info(f"Uploaded shard {shard_id} with {len(shard_data)} items to hf://{hf_repo_id}/{path_in_repo}")
