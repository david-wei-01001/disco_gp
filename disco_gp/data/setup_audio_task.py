"""
Utilities to set up different evaluation/training tasks for DiscoGP.

Supported tasks
---
- IOI (Indirect Object Identification)
- BLiMP (Minimal pair grammaticality)
- PARArel (relational probing)

This module converts raw datasets into HuggingFace `Dataset`s and then into
PyTorch `DataLoader`s for train/eval/test splits according to configuration.
It also provides helper routines for tokenization and for filtering PARArel
examples to those the base model can already answer (so intervention analyses
aren't dominated by unanswerable cases).
"""

import json
from argparse import Namespace

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

label_map = {"bonafide": 0, "spoof": 1}


def split_ratios(ratios):
    """Validate and convert (train, dev, test) ratios into split args.

    Parameters
    ---
    ratios: tuple[float, float, float]
        Expected to sum to 1.0, ordered as (train, dev, test).

    Returns
    ---
    dev_test: float
        Proportion used to carve out *all* evaluation data (dev+test) from the full set.
    test_over_dev: float
        Within the eval split, the fraction allocated to test (the remainder is dev).

    Notes
    ---
    We first split into [train] vs [dev+test], then split [dev+test] into [dev] vs [test].
    This mirrors `Dataset.train_test_split` usage.
    """
    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    assert len(ratios) == 3, "Ratios must be a tuple of three values (train, dev, test)"
    _, dev, test = ratios

    return dev + test, test / (dev + test)


def setup_audio_task(disco_gp, wrapped = None, verbose = False):
    """Dispatch to a specific task setup based on `disco_gp.cfg.task_type`.

    Expected values: {'ioi', 'blimp', 'pararel'}.
    Returns a Namespace with three DataLoaders: train, eval, test.
    """
    if disco_gp.cfg.task_type == 'spoof':
        return setup_spoof(disco_gp, wrapped, verbose)
    elif disco_gp.cfg.task_type == 'blimp':
        return setup_blimp(disco_gp)
    elif disco_gp.cfg.task_type == 'pararel':
        return setup_pararel(disco_gp)
    else:
        raise ValueError(f"Unknown task type: {disco_gp.cfg.task_type}")


def setup_blimp(disco_gp):
    """Prepare BLiMP minimal-pair data as next-token prediction prompts.

    For each BLiMP example we find the longest common prefix between the
    `sentence_good` and `sentence_bad`, then use that as the prompt and treat the
    immediately following tokens as the contrasting targets.
    """
    task = disco_gp.cfg.paradigm
    prompts, targets, targets_good, targets_bad = [], [], [], []

    # Load a specific BLiMP paradigm (e.g., 'anaphor_gender_agreement')
    blimp_ds = load_dataset('blimp', task)
    for row in blimp_ds['train']:
        # Drop trailing period to avoid creating a separate token for '.'
        sg, sb = row['sentence_good'][:-1].split(), row['sentence_bad'][:-1].split()

        combined = []  # longest common prefix accumulator
        target_good, target_bad = None, None
        has_got_full_prefix = False
        for i, (tg, tb) in enumerate(zip(sg, sb)):
            if tg == tb:
                combined.append(tg)
            else:
                # Divergence point: tokens to be predicted
                has_got_full_prefix = True
                target_good, target_bad = tg, tb

            # Until divergence, we keep extending the prefix
            if not has_got_full_prefix:
                continue

        # Construct prompt from common prefix; targets are the next tokens.
        sent = ' '.join(combined)
        prompts.append(sent)
        targets_good.append(' ' + target_good)
        targets_bad.append(' ' + target_bad)
        targets.append((target_good, target_bad))

    # Build a dictionary suitable for a HF Dataset
    data_dict = {}
    data_dict['prompt'] = prompts
    data_dict['targets'] = targets  # for convenience/debugging

    # Tokenize prompts; keep input_ids and per-sequence lengths
    tokenized = disco_gp.tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict['input_ids'] = tokenized['input_ids']
    data_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    # For next-token evaluation we only need the ID of the *first* token in the target strings
    # (which begin with a leading space to trigger correct tokenization boundaries).
    first_token_idx = 0

    data_dict['target good'] = [
        token_ids[first_token_idx] for token_ids in
        disco_gp.tokenizer(targets_good, add_special_tokens=False)['input_ids']
    ]
    data_dict['target bad'] = [
        token_ids[first_token_idx] for token_ids in
        disco_gp.tokenizer(targets_bad, add_special_tokens=False)['input_ids']
    ]

    # Split into train / (dev+test) first, then split latter into dev / test
    dev_test, test_over_dev = split_ratios(disco_gp.cfg.ds_split_ratios)

    ds = Dataset.from_dict(data_dict).train_test_split(dev_test).with_format('torch')
    eval_test_ds = ds['test'].train_test_split(test_over_dev)

    # Build DataLoaders; evaluation loaders are not shuffled.
    train_dl = DataLoader(
        ds['train'],
        batch_size=disco_gp.cfg.batch_size,
    )
    eval_dl = DataLoader(
        eval_test_ds['train'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )
    test_dl = DataLoader(
        eval_test_ds['test'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )

    return Namespace(train=train_dl, eval=eval_dl, test=test_dl)


def setup_spoof(disco_gp, wrapped, verbose: bool = False):
    """
    Minimal loader for LanceaKing/asvspoof2019 that keeps collate as an inner function
    and provides an extensive verbose mode for debugging/inspection.

    Args:
        disco_gp: object with attributes:
            - cfg.batch_size
            - cfg.ds_split_ratios
            - cfg.device (e.g., "cuda" or "cpu")
            - (optional) cfg.verbose (bool) - used if verbose param not provided
        wrapped: A TransformerLens HookedAudioEncoder model
            - to_frames(...) method (as posted earlier)
            - hubert_model attribute used for pos_conv_embed and layer_norm
        verbose: if True, print detailed debug info for each step
                  (if False and disco_gp.cfg.verbose exists, that is used)
    Returns:
        Namespace(train=DataLoader, eval=DataLoader, test=DataLoader)
    """
    # allow cfg override
    if not verbose and hasattr(disco_gp, "cfg") and getattr(disco_gp.cfg, "verbose", None) is not None:
        verbose = bool(disco_gp.cfg.verbose)

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # --- 1) load dataset ---
    try:
        vprint("Loading dataset LanceaKing/asvspoof2019 from Hugging Face...")
        ds = load_dataset("LanceaKing/asvspoof2019", split="train", trust_remote_code=True)
        vprint("Loaded dataset. Number of examples:", len(ds))
        vprint("Dataset column names:", ds.column_names)
    except Exception as e:
        print("ERROR loading dataset LanceaKing/asvspoof2019:", e)
        traceback.print_exc()
        raise

    # --- 2) split into train/dev/test using disco_gp helper ---
    dev_test, test_over_dev = split_ratios(disco_gp.cfg.ds_split_ratios)
    vprint("Splitting dataset with ratios (dev_test, test_over_dev) =", dev_test, test_over_dev)
    ds_split = ds.train_test_split(test_size=dev_test)
    vprint("After first split: train size =", len(ds_split["train"]), "lumped test size =", len(ds_split["test"]))
    eval_test_split = ds_split["test"].train_test_split(test_over_dev)
    vprint("Final split sizes -> train:", len(ds_split["train"]),
           "eval (dev):", len(eval_test_split["train"]),
           "test:", len(eval_test_split["test"]))

    # --- 3) inner collate function with verbose checks ---
    def collate_fn(batch):
        vprint("\n---- collate_fn called ----")
        vprint("Batch size (raw):", len(batch))

        # 3.1 extract waves and sampling rate
        waves = []
        sr = None
        for i, ex in enumerate(batch):
            if "audio" not in ex:
                raise KeyError(f"example {i} has no 'audio' field; keys: {list(ex.keys())}")
            audio = ex["audio"]
            if isinstance(audio, dict) and "array" in audio:
                waves.append(audio["array"])
                if sr is None:
                    sr = audio.get("sampling_rate", None)
            else:
                # fallback: might be path or array-like
                waves.append(audio)
        if sr is None:
            sr = 16000
            vprint("Sampling rate not found in examples; defaulting to", sr)
        vprint("Collected waves:", len(waves), "sampling_rate:", sr)
        lens = [w.shape[0] if hasattr(w, "shape") else len(w) for w in waves]
        vprint("Waveform lengths (samples):", lens)

        # 3.2 transform to frames using user's to_frames
        vprint("Calling disco_gp.to_frames(...) to project to HuBERT frames...")
        try:
            frames, frame_attention_mask = wrapped.to_frames(waves, sampling_rate=sr, move_to_device=False)
        except Exception as e:
            vprint("ERROR inside to_frames(); re-raising after printing traceback.")
            traceback.print_exc()
            raise
        vprint("to_frames returned:")
        vprint("  frames type:", type(frames), "shape:", None if frames is None else tuple(frames.shape),
                "dtype:", None if frames is None else frames.dtype, "device:", frames.device if isinstance(frames, torch.Tensor) else "n/a")
        vprint("  frame_attention_mask type:", type(frame_attention_mask),
                "shape:", None if frame_attention_mask is None else tuple(frame_attention_mask.shape),
                "dtype:", None if frame_attention_mask is None else frame_attention_mask.dtype,
                "device:", None if frame_attention_mask is None else frame_attention_mask.device)

        # 3.3 move frames/mask to target device if needed
        device = disco_gp.cfg.device
        vprint(f"Ensuring frames and mask are on device '{device}' (frames.device = {frames.device})...")
        if frames.device.type != device:
            frames = frames.to(device)
            vprint("  moved frames to", frames.device)
            if frame_attention_mask is not None:
                frame_attention_mask = frame_attention_mask.to(device)
                vprint("  moved frame_attention_mask to", frame_attention_mask.device)
        else:
            vprint("  frames already on target device.")

        # 3.4 positional conv embeddings + resid + layer_norm (inside no_grad since encoder frozen)
        vprint("Computing position embeddings via hubert_model.encoder.pos_conv_embed(...)")
        with torch.no_grad():
            position_embeddings = wrapped.hubert_model.encoder.pos_conv_embed(frames)
            vprint("  position_embeddings shape:", tuple(position_embeddings.shape), "dtype:", position_embeddings.dtype)
            resid = frames + position_embeddings
            vprint("  resid (after add) shape:", tuple(resid.shape))
            resid = wrapped.hubert_model.encoder.layer_norm(resid)
            vprint("  resid (after layer_norm) shape:", tuple(resid.shape))

        # 3.5 build additive attention mask
        vprint("Constructing additive attention mask from frame_attention_mask (1=real, 0=pad expected)...")
        if frame_attention_mask is not None:
            vprint("  raw frame_attention_mask stats -> dtype:", frame_attention_mask.dtype,
                    "min:", frame_attention_mask.min().item(), "max:", frame_attention_mask.max().item(),
                    "sum(valid frames total):", int(frame_attention_mask.sum().item()))
            # pad_mask = 1 where padding, 0 where real
            pad_mask = (1 - frame_attention_mask)  # (B, L)
            vprint("  pad_mask (1=pad) shape:", tuple(pad_mask.shape),
                    "min:", pad_mask.min().item(), "max:", pad_mask.max().item(),
                    "sum(pad positions):", int(pad_mask.sum().item()))
            # expand to (B,1,1,L)
            pad_mask_exp = pad_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,L)
            vprint("  pad_mask_expanded shape:", tuple(pad_mask_exp.shape))
            large_neg = torch.tensor(-1e9, dtype=frames.dtype, device=frames.device)
            additive_attention_mask = pad_mask_exp.to(dtype=frames.dtype) * large_neg
            vprint("  additive_attention_mask shape:", tuple(additive_attention_mask.shape),
                    "min:", float(additive_attention_mask.min().item()),
                    "max:", float(additive_attention_mask.max().item()))
        else:
            additive_attention_mask = None
            vprint("  frame_attention_mask is None -> additive_attention_mask set to None")

        if frame_attention_mask is not None:
            # ensure integer dtype then sum over last dim (L)
            # frame_attention_mask typically has 1 for real, 0 for pad
            seq_lengths = frame_attention_mask.to(torch.long).sum(dim=-1)  # shape (B,)
        else:
            # no mask returned by to_frames -> assume full length (frames shape = B x L x D)
            seq_lengths = torch.full((frames.shape[0],), frames.shape[1], dtype=torch.long)

        vprint("seq_lengths (valid token counts):", seq_lengths.tolist())

        # 3.6 labels
        raw_labels = [ex["label"] for ex in batch]

        # assume dataset label is string; always map with label_map
        label_strs = [str(l) for l in raw_labels]
        label_ids = [label_map[s] for s in label_strs]
        labels = torch.tensor(label_ids, dtype=torch.long, device=frames.device)
        vprint("Labels tensor shape:", tuple(labels.shape), "values:", labels.tolist())
        seq_lengths = seq_lengths.to("cpu")  # keep lengths on CPU by default (commonly useful)

        vprint("collate_fn finished preparing batch. Returning dict with keys: frames, frame_attention_mask, resid, additive_attention_mask, labels")
        return {
            "tokens": resid,                # (B, L, D)
            "seq_length": seq_lengths,      # (B,) long
            "labels": labels,               # (B,) long, mapped {bonafide=0, spoof=1}
            "label_strs": label_strs,       # original string labels
        }

    # --- 4) build dataloaders ---
    vprint("Creating DataLoaders with batch_size =", disco_gp.cfg.batch_size)
    train_dl = DataLoader(
        ds_split["train"],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_dl = DataLoader(
        eval_test_split["train"],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        eval_test_split["test"],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    vprint("DataLoaders created. train batches:", len(train_dl), "eval batches:", len(eval_dl), "test batches:", len(test_dl))

    return Namespace(train=train_dl, eval=eval_dl, test=test_dl)



def setup_ioi_dataset(ioi_prompts, disco_gp):
    """Convert raw IOI prompt dicts to a tokenized HF Dataset.

    Parameters
    ---
    ioi_prompts: list[dict]
        Items contain 'text', 'IO', and 'S'.
    disco_gp: object
        Expected to expose a `tokenizer` compatible with HF tokenizers.
    """
    prompts, targets, io_list, s_list = [], [], [], []
    for item in ioi_prompts:
        prompt_full = item['text']
        # Keep everything up to the last space before the IO token (so model predicts IO)
        prompt = prompt_full[:prompt_full.rfind(' ' + item['IO'])]
        prompts.append(prompt)
        targets.append((item['IO'], item['S']))

        io_list.append(item['IO'])
        s_list.append(item['S'])

    data_dict = {}
    data_dict['prompt'] = prompts
    data_dict['targets'] = targets

    tokenized = disco_gp.tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict['input_ids'] = tokenized['input_ids']
    data_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    # Single-token assumptions: IO and S are expected to be single tokens under the tokenizer.
    data_dict['target good'] = [token_ids[0] for token_ids in disco_gp.tokenizer(io_list)['input_ids']]
    data_dict['target bad'] = [token_ids[0] for token_ids in disco_gp.tokenizer(s_list)['input_ids']]

    ds = Dataset.from_dict(data_dict)
    return ds


def process_pararel_data(disco_gp, ds_dict):
    """Add class indices/vocab for PARArel answers and return a HF Dataset.

    - Tokenizes each answer (without special tokens) and takes the first token id
      as the class label. This assumes single-token answers after a leading space.
    - Stores `answer_idx_vocab` both on `disco_gp` and the returned dataset for later use.
    """
    answer_token_ids = [
        disco_gp.tokenizer(answer, add_special_tokens=False)['input_ids'][0] for answer in ds_dict['answer']
    ]
    # Deduplicate while preserving order via set+sort of numeric ids
    answer_idx_vocab = list(set(answer_token_ids))
    answer_idx_vocab.sort()
    disco_gp.answer_idx_vocab = answer_idx_vocab

    class_idx_list = [answer_idx_vocab.index(x) for x in answer_token_ids]
    ds_dict['answer_idx'] = class_idx_list

    ds = Dataset.from_dict(ds_dict)
    ds.answer_idx_vocab = answer_idx_vocab
    return ds


@torch.no_grad()
def filter_out_unanswerable(disco_gp, ds):
    """Keep only PARArel examples the *base* model answers correctly.

    Rationale
    ---
    For causal analysis/interventions, we often want to avoid examples the base
    model already fails on. This routine runs the current model in eval mode
    (with all masks off) and filters to correct items.

    Side effects
    ---
    - Temporarily turns off weight and edge masks via `disco_gp.turn_off_*` hooks.
    - Attaches `answer_idx_vocab` to the returned dataset for downstream indexing.
    """
    dl = DataLoader(
        ds,
        batch_size=disco_gp.cfg.batch_size,
    )
    # Ensure evaluations reflect the unmodified model
    disco_gp.turn_off_weight_masks()
    disco_gp.turn_off_edge_masks()

    filtered_ds_dict = {
        'prompt': [],
        'answer': [],
        'answer_idx': [],
    }

    for batch in dl:
        bs = torch.arange(len(batch['prompt']))
        batch_input = disco_gp.tokenizer(
            batch['prompt'], return_tensors='pt', padding=True
        ).to(disco_gp.cfg.device)
        lengths = batch_input.attention_mask.sum(dim=1)

        # Forward pass; assume disco_gp(...) returns (logits, *extras)
        logits = disco_gp(batch_input.input_ids)[0]
        # Select the logits at the last context position for each sequence
        # then restrict to the label vocabulary for classification.
        pred_labels = logits[bs, lengths - 1][:, ds.answer_idx_vocab].argmax(-1)
        correctness = (pred_labels.cpu() == batch['answer_idx'])

        for i, correct in enumerate(correctness):
            if correct:
                filtered_ds_dict['prompt'].append(batch['prompt'][i])
                filtered_ds_dict['answer'].append(batch['answer'][i])
                filtered_ds_dict['answer_idx'].append(batch['answer_idx'][i])

    # Cache tokenization for faster later use
    tokenized = disco_gp.tokenizer(filtered_ds_dict['prompt'], return_tensors='pt', padding=True)
    filtered_ds_dict['input_ids'] = tokenized['input_ids']
    filtered_ds_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    new_ds = Dataset.from_dict(filtered_ds_dict)
    new_ds.answer_idx_vocab = ds.answer_idx_vocab
    return new_ds


def setup_pararel(disco_gp):
    """Prepare PARArel dataset and DataLoaders.

    Steps
    ---
    1) Load preprocessed PARArel JSON from `disco_gp.cfg.pararel_data_path`.
    2) Select relation IDs from space-separated string `pararel_rel_ids`.
    3) Build (prompt, answer) pairs; answers are prefixed with a leading space.
    4) Convert to HF Dataset with class indices via `process_pararel_data`.
    5) Filter out examples the base model cannot answer via `filter_out_unanswerable`.
    6) Split into train/dev/test and wrap in DataLoaders.
    """

    # 1) Load raw relation data (a dict: rel_id -> list of entries)
    with open(disco_gp.cfg.pararel_data_path) as open_file:
        pararel_rel_data = json.load(open_file)

    # 2) Which relations to include (space-separated in config)
    rel_ids = disco_gp.cfg.pararel_rel_ids.split(' ')

    # Collect all entries across the chosen relations
    data = []
    for rel_id in rel_ids:
        data += pararel_rel_data[rel_id]

    # 3) Build prompt/answer fields from the PARArel template entries
    ds_dict = {
        'prompt': [],
        'answer': [],
    }
    for entry in data:
        # entry[0] is usually a (template, answer) pair. Strip the trailing "[MASK] ." variations.
        prompt = entry[0][0].replace(' [MASK] .', '')
        prompt = prompt.replace(' [MASK].', '')
        assert '[MASK]' not in prompt
        target = entry[0][1]
        ds_dict['prompt'].append(prompt)
        ds_dict['answer'].append(' ' + target)  # leading space for tokenizer boundary

    # 4) Add class indices and vocab
    ds = process_pararel_data(disco_gp, ds_dict)

    # 5) Only keep examples that the base model can answer correctly
    ds = filter_out_unanswerable(disco_gp, ds)

    # 6) Split and wrap in loaders
    dev_test, test_over_dev = split_ratios(disco_gp.cfg.ds_split_ratios)

    ds = ds.train_test_split(dev_test).with_format('torch')
    eval_test_ds = ds['test'].train_test_split(test_over_dev)

    train_dl = DataLoader(
        ds['train'],
        batch_size=disco_gp.cfg.batch_size,
    )
    eval_dl = DataLoader(
        eval_test_ds['train'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )
    test_dl = DataLoader(
        eval_test_ds['test'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )

    return Namespace(train=train_dl, eval=eval_dl, test=test_dl)
