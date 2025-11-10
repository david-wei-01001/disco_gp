# inspect_asvspoof.py
import os
import glob
import pandas as pd
import soundfile as sf    # pip install soundfile
from pathlib import Path

DATA_DIR = "asvspoof2019"  # change if different

def find_text_protocols(root):
    # look for common protocol/trl files or any txt files
    patterns = ["**/*protocol*.txt", "**/*trl*.txt", "**/*.txt"]
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(root, p), recursive=True)
    # remove duplicates and sort by name/size (heuristic)
    files = sorted(set(files), key=lambda f: (-os.path.getsize(f), f))
    return files

def load_protocol_table(path):
    # read whitespace-separated lines; handle variable column counts
    with open(path, "r", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    parts = [ln.split() for ln in lines]
    max_cols = max(len(p) for p in parts)
    df = pd.DataFrame([p + [""]*(max_cols - len(p)) for p in parts])
    return df

def find_audio_for_basename(root, base):
    # search for files that contain the base name (robust for different extensions)
    pat = os.path.join(root, "**", f"{base}*")
    matches = glob.glob(pat, recursive=True)
    # prefer wav files
    wavs = [m for m in matches if m.lower().endswith(".wav")]
    return wavs[0] if wavs else (matches[0] if matches else None)

def main():
    root = DATA_DIR
    if not os.path.isdir(root):
        print("ERROR: data dir not found:", root)
        return

    print("Searching for protocol/metadata text files under:", root)
    protocols = find_text_protocols(root)
    for i, p in enumerate(protocols[:10], 1):
        print(f"{i}. {p}  (size {os.path.getsize(p):,} bytes)")
    if not protocols:
        print("No protocol txt found. Show tree of top-level folders:")
        for entry in os.listdir(root):
            print(" ", entry)
        return

    # choose the largest protocol file (heuristic)
    proto = protocols[0]
    print("\nParsing protocol file (chosen):", proto)
    df = load_protocol_table(proto)
    print("Protocol table shape:", df.shape)
    print("First 7 rows (raw columns):")
    print(df.head(7))

    # The dataset usually puts the label in the last column; print unique label values:
    possible_label_col = df.columns[-1]
    unique_labels = df[possible_label_col].unique()
    print(f"\nUnique values in last column (col index {possible_label_col}):")
    print(unique_labels)
    print("\nCounts:")
    print(df[possible_label_col].value_counts())

    # show a small sample and try to locate the audio files
    print("\nSample rows and attempt to locate matching audio files (if present):")
    sample = df.sample(min(8, len(df)), random_state=42)
    for idx, row in sample.iterrows():
        # pick the first column as filename candidate, but handle if it contains path/extension
        candidate = str(row.iloc[0])
        base = os.path.splitext(os.path.basename(candidate))[0]
        audio_path = find_audio_for_basename(root, base)
        if audio_path:
            try:
                info = sf.info(audio_path)
                duration = info.frames / info.samplerate
                print(f"{base}: found -> {audio_path}  sr={info.samplerate} dur={duration:.3f}s  label={row.iloc[-1]}")
            except Exception as e:
                print(f"{base}: found -> {audio_path}  (error reading with soundfile: {e}) label={row.iloc[-1]}")
        else:
            print(f"{base}: NO audio file found in {root}  label={row.iloc[-1]}")

    # If you want a binary mapping (example):
    lbls = df[possible_label_col].astype(str).str.lower()
    # simple heuristic mapping:
    binary = lbls.map(lambda s: 0 if ("bonafide" in s or "genuine" in s or "target" in s) else (1 if ("spoof" in s or "attack" in s) else None))
    print("\nBinary mapping counts (0=bonafide/genuine, 1=spoof/attack, None=unknown):")
    print(binary.value_counts(dropna=False))

if __name__ == "__main__":
    main()
