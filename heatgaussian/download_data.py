from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO_ID = "desmondlzy/higs-data"
DEFAULT_DATA_ROOT = Path("data")


def maybe_download_data(
    data_path: str | Path,
    hf_repo_id: str = HF_REPO_ID,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> Path:

    data_path = Path(data_path)
    data_root = Path(data_root).resolve()

    if data_path.exists():
        return data_path

    # the relative path need to in the repo
    try:
        rel = data_path.resolve().relative_to(data_root)
    except ValueError:
        # data_path is not under data_root – can't map to a HF sub-path
        raise FileNotFoundError(
            f"Data path '{data_path}' does not exist and is not under '{data_root}'"
        )

    print(f"Data path '{data_path}' not found locally.")
    print(f"Downloading '{rel}' from HuggingFace repo '{hf_repo_id}' ...")

    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="dataset",
        allow_patterns=[f"{rel}/**"],
        local_dir=str(data_root),
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Download completed but '{data_path}' still not found. "
        )

    print(f"Download complete: {data_path}")
    return data_path
