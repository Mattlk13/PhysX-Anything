from huggingface_hub import snapshot_download

local_dir_our = snapshot_download(repo_id="Caoza/PhysX-Anything", local_dir="pretrain")

local_dir = snapshot_download(repo_id="microsoft/TRELLIS-image-large", local_dir="pretrain/trellis")

