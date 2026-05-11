import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
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


def github_api(
    method: str,
    url: str,
    token: str,
    data: bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, bytes]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if data is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def ensure_release(repo: str, tag: str, token: str) -> dict:
    status, body = github_api(
        "GET",
        f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
        token,
    )
    if status == 200:
        return json.loads(body)
    payload = json.dumps(
        {
            "tag_name": tag,
            "name": f"Colab synced model artifacts ({tag})",
            "body": "Automatically uploaded Applio/RVC model artifacts from Colab.",
        }
    ).encode("utf-8")
    status, body = github_api(
        "POST",
        f"https://api.github.com/repos/{repo}/releases",
        token,
        payload,
    )
    if status not in {200, 201}:
        raise RuntimeError(f"Could not create GitHub release {repo}@{tag}: {status} {body[:500]!r}")
    return json.loads(body)


def upload_release_asset(repo: str, tag: str, token: str, path: Path) -> None:
    release = ensure_release(repo, tag, token)
    upload_url = release["upload_url"].split("{", 1)[0]
    asset_name = path.name
    existing_assets = release.get("assets", [])
    for asset in existing_assets:
        if asset.get("name") == asset_name:
            github_api("DELETE", asset["url"], token)
            break

    url = f"{upload_url}?name={urllib.parse.quote(asset_name)}"
    print(f"Uploading GitHub Release asset: {asset_name}", flush=True)
    status, body = github_api(
        "POST",
        url,
        token,
        path.read_bytes(),
        content_type="application/octet-stream",
    )
    if status not in {200, 201}:
        raise RuntimeError(f"Upload failed for {asset_name}: {status} {body[:500]!r}")
    print(f"Uploaded GitHub Release asset: {asset_name}", flush=True)


def monitor_and_upload_exports(
    log_dir: Path,
    stop_event: threading.Event,
    repo: str,
    tag: str,
    token: str,
    poll_seconds: int = 60,
) -> None:
    uploaded: set[str] = set()
    while not stop_event.is_set():
        candidates = sorted(
            path
            for path in log_dir.glob("*.pth")
            if path.is_file()
            and not path.name.startswith("G_")
            and not path.name.startswith("D_")
            and path.name not in uploaded
        )
        for path in candidates:
            try:
                # Wait for the file to settle before uploading.
                size_a = path.stat().st_size
                time.sleep(3)
                size_b = path.stat().st_size
                if size_a != size_b or size_b == 0:
                    continue
                upload_release_asset(repo, tag, token, path)
                uploaded.add(path.name)
            except Exception as exc:
                print(f"GitHub sync failed for {path.name}: {exc}", flush=True)
        stop_event.wait(poll_seconds)


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
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--github-repo", default="tomjiu/isekaijoucho_colab_repo")
    parser.add_argument("--github-tag", default="colab-model-backups")
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

    train_cmd = [
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
    ]

    stop_event = threading.Event()
    monitor_thread = None
    if args.github_token:
        ensure_release(args.github_repo, args.github_tag, args.github_token)
        monitor_thread = threading.Thread(
            target=monitor_and_upload_exports,
            args=(log_dir, stop_event, args.github_repo, args.github_tag, args.github_token),
            daemon=True,
        )
        monitor_thread.start()
        print(
            f"GitHub model backup enabled: https://github.com/{args.github_repo}/releases/tag/{args.github_tag}",
            flush=True,
        )
    else:
        print("GitHub model backup disabled because no token was provided.", flush=True)

    try:
        run(train_cmd, cwd=applio_dir)
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=120)

    if args.github_token:
        for path in sorted(
            path
            for path in log_dir.glob("*.pth")
            if path.is_file() and not path.name.startswith("G_") and not path.name.startswith("D_")
        ):
            upload_release_asset(args.github_repo, args.github_tag, args.github_token, path)

    output_zip = Path("/content/isekaijoucho_colab_outputs.zip")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in log_dir.rglob("*"):
            if path.is_file() and path.suffix in {".pth", ".index", ".json", ".txt"}:
                archive.write(path, path.relative_to(applio_dir))
    print(f"\nOutput zip: {output_zip}", flush=True)


if __name__ == "__main__":
    main()
