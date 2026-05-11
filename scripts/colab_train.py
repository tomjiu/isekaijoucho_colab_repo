import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, dst)


def unzip(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {src} -> {dst}", flush=True)
    with zipfile.ZipFile(src) as archive:
        archive.extractall(dst)


def install_applio(applio_dir: Path) -> None:
    if not applio_dir.exists():
        run(["git", "clone", "https://github.com/iahispano/applio", str(applio_dir)])
    run([sys.executable, "-m", "pip", "install", "-U", "pip", "uv"])
    run(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "-r",
            "requirements.txt",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
            "--index-strategy",
            "unsafe-best-match",
        ],
        cwd=applio_dir,
    )
    run([sys.executable, "-m", "pip", "install", "faster-whisper", "imageio-ffmpeg", "webrtcvad"])
    run(
        [
            sys.executable,
            "core.py",
            "prerequisites",
            "--pretraineds_hifigan",
            "True",
            "--models",
            "True",
            "--exe",
            "False",
        ],
        cwd=applio_dir,
    )


def copy_prepare_script(repo_dir: Path, applio_dir: Path) -> None:
    src = repo_dir / "scripts" / "prepare_isekaijoucho_dataset.py"
    dst = applio_dir / "scripts" / "prepare_isekaijoucho_dataset.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def patch_prepare_paths(script_path: Path, source_dir: Path, output_dir: Path) -> None:
    text = script_path.read_text(encoding="utf-8")
    text = text.replace(
        'SOURCE_DIR = Path(r"E:\\Code\\1\\isekaijoucho")',
        f'SOURCE_DIR = Path(r"{source_dir}")',
    )
    text = text.replace(
        'OUTPUT_DIR = Path(r"E:\\Code\\ai\\applio_isekaijoucho_local\\datasets\\isekaijoucho_autotrim")',
        f'OUTPUT_DIR = Path(r"{output_dir}")',
    )
    script_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="/content/isekaijoucho_colab_repo")
    parser.add_argument("--applio-dir", default="/content/Applio")
    parser.add_argument("--data-url", required=True)
    parser.add_argument("--checkpoint-url", default="")
    parser.add_argument("--model-name", default="isekaijoucho_whispertrim")
    parser.add_argument("--total-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=40000)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--cpu-cores", type=int, default=4)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    applio_dir = Path(args.applio_dir)
    source_dir = Path("/content/isekaijoucho_source_audio")
    dataset_dir = applio_dir / "datasets" / "isekaijoucho_autotrim"
    log_dir = applio_dir / "logs" / args.model_name

    install_applio(applio_dir)

    data_zip = Path("/content/isekaijoucho_source_audio.zip")
    download(args.data_url, data_zip)
    unzip(data_zip, source_dir)

    copy_prepare_script(repo_dir, applio_dir)
    prepare_script = applio_dir / "scripts" / "prepare_isekaijoucho_dataset.py"
    patch_prepare_paths(prepare_script, source_dir, dataset_dir)
    run([sys.executable, str(prepare_script)], cwd=applio_dir)

    run(
        [
            sys.executable,
            "core.py",
            "preprocess",
            "--model_name",
            args.model_name,
            "--dataset_path",
            str(dataset_dir),
            "--sample_rate",
            str(args.sample_rate),
            "--cpu_cores",
            str(args.cpu_cores),
            "--cut_preprocess",
            "Automatic",
            "--process_effects",
            "False",
            "--noise_reduction",
            "False",
            "--noise_reduction_strength",
            "0.0",
            "--chunk_len",
            "3.0",
            "--overlap_len",
            "0.3",
            "--normalization_mode",
            "post",
        ],
        cwd=applio_dir,
    )

    run(
        [
            sys.executable,
            "core.py",
            "extract",
            "--model_name",
            args.model_name,
            "--f0_method",
            args.f0_method,
            "--cpu_cores",
            str(args.cpu_cores),
            "--gpu",
            "0",
            "--sample_rate",
            str(args.sample_rate),
            "--embedder_model",
            "contentvec",
            "--include_mutes",
            "2",
        ],
        cwd=applio_dir,
    )

    if args.checkpoint_url:
        checkpoint_zip = Path("/content/isekaijoucho_epoch1_checkpoint.zip")
        download(args.checkpoint_url, checkpoint_zip)
        unzip(checkpoint_zip, log_dir)

    run(
        [
            sys.executable,
            "core.py",
            "train",
            "--model_name",
            args.model_name,
            "--vocoder",
            "HiFi-GAN",
            "--checkpointing",
            "True",
            "--save_every_epoch",
            "1",
            "--save_only_latest",
            "False",
            "--save_every_weights",
            "True",
            "--total_epoch",
            str(args.total_epoch),
            "--sample_rate",
            str(args.sample_rate),
            "--batch_size",
            str(args.batch_size),
            "--gpu",
            "0",
            "--pretrained",
            "True",
            "--overtraining_detector",
            "True",
            "--overtraining_threshold",
            "10",
            "--cleanup",
            "False",
            "--cache_data_in_gpu",
            "False",
            "--index_algorithm",
            "Auto",
        ],
        cwd=applio_dir,
    )

    output_zip = Path("/content/isekaijoucho_colab_outputs.zip")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in log_dir.rglob("*"):
            if path.is_file() and path.suffix in {".pth", ".index", ".json", ".txt"}:
                archive.write(path, path.relative_to(applio_dir))
    print(f"\nOutput zip: {output_zip}", flush=True)


if __name__ == "__main__":
    main()
