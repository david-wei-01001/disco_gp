source ~/CNoiSY/myenv/bin/activate
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_DATASETS_DOWNLOAD_TIMEOUT=120

# Retries
export HF_HUB_DOWNLOAD_RETRIES=20
export HF_DATASETS_DOWNLOAD_RETRIES=20

# Optional: parallel threads (avoid too many concurrent HTTPS requests)
export HF_DATASETS_DOWNLOAD_NUM_PROC=4
# set Hugging Face token
export HF_ACCESS_TOKEN=
export HF_HUB_ENABLE_HF_TRANSFER=1

python3 audio_run.py
