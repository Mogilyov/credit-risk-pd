"""Download raw dataset for Credit Risk PD project from Kaggle competition."""

import subprocess
import zipfile
from pathlib import Path

import kaggle


def download_data(out_dir: str = "data/raw") -> None:
    """Download Give Me Some Credit dataset from Kaggle competition and unzip it.

    Note: Requires kaggle CLI to be installed and configured with API credentials.
    Run 'pip install kaggle' and place kaggle.json in ~/.kaggle/ if not done yet.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    zip_path = out_path / "GiveMeSomeCredit.zip"

    print("Downloading Give Me Some Credit dataset from Kaggle...")
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "GiveMeSomeCredit",
                "-p",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Dataset downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e.stderr}")
        raise SystemExit(1)
    except FileNotFoundError:
        print(
            "Error: kaggle CLI not found. Please install it with 'pip install kaggle'"
        )
        raise SystemExit(1)

    # Unzip the archive
    if zip_path.exists():
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(out_path)
        print("Unzipping completed.")
        zip_path.unlink()  # Optionally remove the zip file after extraction
    else:
        print(f"Archive {zip_path} not found after download!")
        raise SystemExit(1)


def download_dataset():
    """Download the RESD dataset from Kaggle to the data directory."""
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "gauravduttakiit/resd-dataset",
        path="data",
        unzip=True,
        force=False,
    )


if __name__ == "__main__":
    download_data()
