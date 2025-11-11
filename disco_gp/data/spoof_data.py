# spoof_data.py
"""
Utilities to load ASVspoof2019 LA (local copy) and create DataLoaders.

Functions:
  - load_spoof_data(data_dir) -> datasets.Dataset with columns:
        'audio_path' (str), 'label' (int), 'file_id' (str), 'orig_split' (str)
  - make_spoof_dataloaders(ds, batch_size, feature_extractor=None, ds_split_ratios=(0.8,0.1,0.1), target_sr=16000)
        -> argparse.Namespace(train=DataLoader, eval=DataLoader, test=DataLoader)

CLI:
  python spoof_data.py /path/to/LA --batch-size 8 --preview

Notes:
  - Expects folder structure like:
      /.../disco_data/LA/ASVspoof2019_LA_cm_protocols
      /.../disco_data/LA/ASVspoof2019_LA_train
      /.../disco_data/LA/ASVspoof2019_LA_dev
      /.../disco_data/LA/ASVspoof2019_LA_eval
  - Uses soundfile to read audio (pip install soundfile)
  - Uses transformers.AutoFeatureExtractor if not provided.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import re
import logging

import torch
from torch.utils.data import DataLoader
from argparse import Namespace
from collections import Counter
from datasets import Dataset

# third-party libraries
from datasets import Dataset
import soundfile as sf
from transformers import AutoFeatureExtractor

LABEL_MAP = {"bonafide": 0, "spoof": 1}
DATA_DIR = 

# Map splits to audio folders
AUDIO_SPLITS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}

# --- robust parser for the protocol lines ---
LA_ID_REGEX = re.compile(r"LA_[A-Z]_[0-9]+", flags=re.I)  # matches LA_T_123 etc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_audio_task(disco_gp = None, wrapped_model = None):
    """Dispatch to a specific task setup based on `disco_gp.cfg.task_type`.

    Expected values: {'ioi', 'blimp', 'pararel'}.
    Returns a Namespace with three DataLoaders: train, eval, test.
    """
    return setup_spoof(disco_gp, wrapped_model)
    # if disco_gp.cfg.task_type == 'spoof':
    #     return setup_spoof(disco_gp, wrapped_model)
    # else:
    #     raise ValueError(f"Unknown task type: {disco_gp.cfg.task_type}")

# add to spoof_data.py

def setup_spoof(disco_gp = None, wrapped_model = None):
    """
    Prepare ASVspoof2019-LA dataloaders for DiscoGP.

    Expects disco_gp.cfg to contain:
      - data_dir (str): path to ASVspoof2019 LA root
      - batch_size (int)
      - ds_split_ratios (tuple of 3 floats) optional, default (0.8,0.1,0.1)
      - feature_extractor (optional): a transformers AutoFeatureExtractor instance
      - target_sr (int) optional, default 16000
      - num_workers (int) optional, default 2

    Returns:
      Namespace(train=DataLoader, eval=DataLoader, test=DataLoader)
    """
    # read config with reasonable defaults
    cfg = getattr(disco_gp, "cfg", None)
    if cfg is None:
        print("no config passed")
        assert wrapped_model is None

    data_dir = DATA_DIR
    batch_size = getattr(cfg, "batch_size", 8)
    ds_split_ratios = getattr(cfg, "ds_split_ratios", (0.8, 0.1, 0.1))
    target_sr = getattr(cfg, "target_sr", 16000)
    num_workers = getattr(cfg, "num_workers", 2)

    # 1) build HF-style dataset listing audio paths / labels
    ds = load_spoof_data(data_dir)

    # 2) build DataLoaders (this will create feature_extractor if None)
    return make_spoof_dataloaders(
        ds,
        wrapped=wrapped_model,
        batch_size=batch_size,
        ds_split_ratios=ds_split_ratios,
        target_sr=target_sr,
        num_workers=num_workers,
    )

# --- helper: list protocol files and guess split by name ---
def _find_cm_protocol_files(cm_proto_dir: str) -> List[Tuple[str, str]]:
    """Return list of (proto_path, split_hint)"""
    files = glob.glob(os.path.join(cm_proto_dir, "*.txt"))
    pairs = []
    for f in sorted(files):
        name = os.path.basename(f).lower()
        if "train" in name or ".trn" in name:
            pairs.append((f, "train"))
        elif "dev" in name or ".dev" in name:
            pairs.append((f, "dev"))
        elif "eval" in name or ".eval" in name:
            pairs.append((f, "eval"))
        else:
            # fallback: keep but mark unknown
            pairs.append((f, "unknown"))
    return pairs

def _parse_cm_protocol(path: str, split_hint: str) -> List[Tuple[str, int, str]]:
    """
    Return list of tuples (file_id, label_int, split_hint).
    Robustly finds the token that looks like the file/utterance id (e.g. LA_T_1000137)
    and the token that is the label (bonafide/spoof).
    """
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            # find label token
            label_idx = None
            for i, p in enumerate(parts):
                if p.lower() in LABEL_MAP:
                    label_idx = i
                    break
            if label_idx is None:
                # no known label in this line -> skip
                continue
            label_token = parts[label_idx].lower()
            label_int = LABEL_MAP[label_token]

            # find the best candidate token for file_id:
            file_token = None
            # 1) any token matching LA_*_\d+ regex
            for p in parts:
                if LA_ID_REGEX.search(p):
                    file_token = p
                    break
            # 2) fallback: token immediately before label (common layout)
            if file_token is None and label_idx >= 1:
                file_token = parts[label_idx - 1]
            # 3) fallback: second token (many protocols have index speaker file)
            if file_token is None and len(parts) >= 3:
                file_token = parts[2]
            # 4) last resort: first token
            if file_token is None and parts:
                file_token = parts[0]

            # normalize: basename + strip extension
            file_id = os.path.splitext(os.path.basename(file_token))[0]
            rows.append((file_id, label_int, split_hint))
    return rows

# --- flexible audio file lookup ---
def _find_audio_for_id(audio_root: str, file_id: str) -> Optional[str]:
    """
    Try to locate an audio file for `file_id` inside audio_root.
    Will look in flac/ and root and fall back to recursive glob.
    """
    base = os.path.splitext(os.path.basename(file_id))[0]

    # search priority locations / extensions
    exts = ("flac", "wav", "mp3", "m4a")
    subpaths = [os.path.join(audio_root, "flac"), audio_root]

    for sub in subpaths:
        for ext in exts:
            cand = os.path.join(sub, f"{base}.{ext}")
            if os.path.exists(cand):
                return cand

    # last resort - recursive glob for anything starting with base
    pattern = os.path.join(audio_root, "**", f"{base}.*")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    return None

# --- main loader ---
def load_spoof_data(data_dir: str = DATA_DIR, require_any: bool = True) -> Dataset:
    """
    Load ASVspoof2019 LA protocol files and return a HuggingFace Dataset
    with columns: audio_path (str), label (int), file_id (str), orig_split (str).
    - require_any: if True raise error when nothing found, else return empty Dataset.
    """
    cm_proto_dir = os.path.join(data_dir, "ASVspoof2019_LA_cm_protocols")
    if not os.path.exists(cm_proto_dir):
        raise FileNotFoundError(f"protocol dir not found: {cm_proto_dir!s}")

    proto_files = _find_cm_protocol_files(cm_proto_dir)
    logger.info("Protocol files found: %d", len(proto_files))
    for p, s in proto_files:
        logger.info("  %s -> %s", p, s)

    audio_paths, labels, file_ids, splits = [], [], [], []
    missing = 0
    for proto_path, split_hint in proto_files:
        parsed = _parse_cm_protocol(proto_path, split_hint)
        logger.info("Parsed %d entries from %s", len(parsed), os.path.basename(proto_path))
        for file_id, lbl, split in parsed:
            # map split token to actual folder; if mapping fails, use split as-is
            audio_root = os.path.join(data_dir, AUDIO_SPLITS.get(split, split))
            audio_path = _find_audio_for_id(audio_root, file_id)
            if audio_path is None:
                missing += 1
                # optional: uncomment next line to debug which ids are missing
                # logger.debug("Missing audio for %s under %s", file_id, audio_root)
                continue
            audio_paths.append(audio_path)
            labels.append(lbl)
            file_ids.append(file_id)
            splits.append(split)

    logger.info("Total found audio: %d (missing %d entries)", len(audio_paths), missing)
    if len(audio_paths) == 0 and require_any:
        raise ValueError(
            "No audio files matched protocol entries. "
            f"Checked {len(proto_files)} protocol files in {cm_proto_dir}."
        )

    ds = Dataset.from_dict({
        "audio_path": audio_paths,
        "label": labels,
        "file_id": file_ids,
        "orig_split": splits
    })

    # quick sanity prints
    counts = Counter(ds["orig_split"])
    logger.info("Examples per split: %s", dict(counts))
    # show a few examples
    for i in range(min(5, len(ds))):
        logger.info("Example %d: %s -> %s (label=%d)", i, ds[i]["file_id"], ds[i]["audio_path"], ds[i]["label"])

    return ds
def split_ratios(ratios):
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-9, "ratios must be (train,dev,test) summing to 1.0"
    train, dev, test = ratios
    dev_test = dev + test
    test_over_dev = 0.0
    if dev_test > 0:
        test_over_dev = test / dev_test
    return dev_test, test_over_dev


def make_spoof_dataloaders(
    ds,
    *,
    wrapped = None,
    batch_size: int = 8,
    ds_split_ratios: Tuple[float,float,float] = (0.8, 0.1, 0.1),
    target_sr: int = 16000,
    num_workers: int = 2,
):
    """
    ds: HF datasets.Dataset with 'audio_path' and 'label' columns.
    Returns Namespace(train=DataLoader, eval=DataLoader, test=DataLoader).
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    if "audio_path" not in ds.column_names or "label" not in ds.column_names:
        raise ValueError("Dataset must have 'audio_path' and 'label' columns")

    dev_test, test_over_dev = split_ratios(ds_split_ratios)
    if dev_test <= 0:
        raise ValueError("ds_split_ratios must reserve some eval data (dev+test) > 0")

    # train vs (dev+test)
    ds_train_hold = ds.train_test_split(test_size=dev_test, seed=42)
    train_ds = ds_train_hold["train"]
    eval_test_ds = ds_train_hold["test"]

    # split eval_test into eval vs test
    eval_test_split = eval_test_ds.train_test_split(test_size=test_over_dev, seed=43)
    eval_ds = eval_test_split["train"]
    test_ds = eval_test_split["test"]

    def collate_fn(batch):
        waveforms = []
        labels = []
        for row in batch:
            p = row["audio_path"]
            try:
                wav, sr = sf.read(p, dtype="float32")
            except Exception as e:
                raise RuntimeError(f"Error reading {p}: {e}")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)  # to mono
            # resample if needed and torchaudio available
            if sr != target_sr:
                try:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, len)
                    wav = resampler(wav_t).squeeze(0).numpy()
                    sr = target_sr
                except Exception:
                    warnings.warn(f"File {p} has sample rate {sr} != {target_sr} but torchaudio not available; returning raw samples.")
            waveforms.append(wav)
            labels.append(int(row["label"]))

        enc = feature_extractor(waveforms, sampling_rate=target_sr, padding=True, return_tensors="pt")
        input_values = enc["input_values"]           # (B, L)
        attention_mask = enc.get("attention_mask")   # (B, L) or None
        labels_t = torch.tensor(labels, dtype=torch.float32)
        return {
            "tokens": input_values,                # (B, L, D)
            "attention_mask": attention_mask,      # (B,) long
            "labels": labels_t,               # (B,) long, mapped {bonafide=0, spoof=1}
        }

    def wrapped_collate_fn(batch):
        logger.info("\n---- collate_fn called ----")
        logger.info("Batch size (raw):", len(batch))

        # 3.1 extract waves and sampling rate
        waves = []
        sr = None
        for i, ex in enumerate(batch):
            audio = ex["audio_path"]
                
            if isinstance(audio, dict) and "array" in audio:
                waves.append(audio["array"])
                if sr is None:
                    sr = audio.get("sampling_rate", None)
            else:
                # fallback: might be path or array-like
                waves.append(audio)
        if sr is None:
            sr = 16000
            logger.info("Sampling rate not found in examples; defaulting to", sr)
        logger.info("Collected waves:", len(waves), "sampling_rate:", sr)
        lens = [w.shape[0] if hasattr(w, "shape") else len(w) for w in waves]
        logger.info("Waveform lengths (samples):", lens)

        # 3.2 transform to frames using user's to_frames
        logger.info("Calling disco_gp.to_frames(...) to project to HuBERT frames...")
        frames, frame_attention_mask = wrapped.to_frames(waves, sampling_rate=sr, move_to_device=False)
        logger.info("to_frames returned:")
        logger.info("  frames type:", type(frames), "shape:", None if frames is None else tuple(frames.shape),
                "dtype:", None if frames is None else frames.dtype, "device:", frames.device if isinstance(frames, torch.Tensor) else "n/a")
        logger.info("  frame_attention_mask type:", type(frame_attention_mask),
                "shape:", None if frame_attention_mask is None else tuple(frame_attention_mask.shape),
                "dtype:", None if frame_attention_mask is None else frame_attention_mask.dtype,
                "device:", None if frame_attention_mask is None else frame_attention_mask.device)

        # 3.3 move frames/mask to target device if needed
        device = disco_gp.cfg.device
        logger.info(f"Ensuring frames and mask are on device '{device}' (frames.device = {frames.device})...")
        if frames.device.type != device:
            frames = frames.to(device)
            logger.info("  moved frames to", frames.device)
            if frame_attention_mask is not None:
                frame_attention_mask = frame_attention_mask.to(device)
                logger.info("  moved frame_attention_mask to", frame_attention_mask.device)
        else:
            logger.info("  frames already on target device.")

        # 3.4 positional conv embeddings + resid + layer_norm (inside no_grad since encoder frozen)
        logger.info("Computing position embeddings via hubert_model.encoder.pos_conv_embed(...)")
        with torch.no_grad():
            position_embeddings = wrapped.hubert_model.encoder.pos_conv_embed(frames)
            logger.info("  position_embeddings shape:", tuple(position_embeddings.shape), "dtype:", position_embeddings.dtype)
            resid = frames + position_embeddings
            logger.info("  resid (after add) shape:", tuple(resid.shape))
            resid = wrapped.hubert_model.encoder.layer_norm(resid)
            logger.info("  resid (after layer_norm) shape:", tuple(resid.shape))

        # 3.5 build additive attention mask
        logger.info("Constructing additive attention mask from frame_attention_mask (1=real, 0=pad expected)...")
        if frame_attention_mask is not None:
            logger.info("  raw frame_attention_mask stats -> dtype:", frame_attention_mask.dtype,
                    "min:", frame_attention_mask.min().item(), "max:", frame_attention_mask.max().item(),
                    "sum(valid frames total):", int(frame_attention_mask.sum().item()))
            # pad_mask = 1 where padding, 0 where real
            pad_mask = (1 - frame_attention_mask)  # (B, L)
            logger.info("  pad_mask (1=pad) shape:", tuple(pad_mask.shape),
                    "min:", pad_mask.min().item(), "max:", pad_mask.max().item(),
                    "sum(pad positions):", int(pad_mask.sum().item()))
            # expand to (B,1,1,L)
            pad_mask_exp = pad_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,L)
            logger.info("  pad_mask_expanded shape:", tuple(pad_mask_exp.shape))
            large_neg = torch.tensor(-1e9, dtype=frames.dtype, device=frames.device)
            additive_attention_mask = pad_mask_exp.to(dtype=frames.dtype) * large_neg
            logger.info("  additive_attention_mask shape:", tuple(additive_attention_mask.shape),
                    "min:", float(additive_attention_mask.min().item()),
                    "max:", float(additive_attention_mask.max().item()))
        else:
            additive_attention_mask = None
            logger.info("  frame_attention_mask is None -> additive_attention_mask set to None")

        if frame_attention_mask is not None:
            # ensure integer dtype then sum over last dim (L)
            # frame_attention_mask typically has 1 for real, 0 for pad
            seq_lengths = frame_attention_mask.to(torch.long).sum(dim=-1)  # shape (B,)
        else:
            # no mask returned by to_frames -> assume full length (frames shape = B x L x D)
            seq_lengths = torch.full((frames.shape[0],), frames.shape[1], dtype=torch.long)

        logger.info("seq_lengths (valid token counts):", seq_lengths.tolist())

        # 3.6 labels
        raw_labels = [ex["label"] for ex in batch]

        # assume dataset label is string; always map with label_map
        label_strs = [str(l) for l in raw_labels]
        label_ids = [label_map[s] for s in label_strs]
        labels = torch.tensor(label_ids, dtype=torch.long, device=frames.device)
        logger.info("Labels tensor shape:", tuple(labels.shape), "values:", labels.tolist())
        seq_lengths = seq_lengths.to("cpu")  # keep lengths on CPU by default (commonly useful)

        logger.info("collate_fn finished preparing batch. Returning dict with keys: frames, frame_attention_mask, resid, additive_attention_mask, labels")
        return {
            "tokens": resid,                # (B, L, D)
            "seq_length": seq_lengths,      # (B,) long
            "labels": labels,               # (B,) long, mapped {bonafide=0, spoof=1}
            "label_strs": label_strs,       # original string labels
        }
    if wrapped is None:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
        eval_dl  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=wrapped_collate_fn, num_workers=num_workers)
        eval_dl  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn, num_workers=num_workers)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=wrapped_collate_fn, num_workers=num_workers)

    return Namespace(train=train_dl, eval=eval_dl, test=test_dl)
