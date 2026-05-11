# isekaijoucho Applio Colab Trainer

Colab helper for training the `isekaijoucho_whispertrim` Applio/RVC model from the local source audio.

## What Goes In Git

- `notebooks/isekaijoucho_applio_colab.ipynb`: Colab notebook launcher.
- `scripts/colab_train.py`: end-to-end Colab runner.
- `scripts/prepare_isekaijoucho_dataset.py`: Whisper-based intro trimming script.

Large audio/checkpoint files should be uploaded as GitHub Release assets, not committed to git.

## Release Assets

Recommended assets:

- `isekaijoucho_source_audio.zip`: original MP3/WAV source files.
- `isekaijoucho_epoch1_checkpoint.zip`: optional local epoch-1 resume checkpoints.
- `isekaijoucho_preprocessed_logs.zip`: optional preprocessed Applio logs to skip Colab CPU preprocessing and feature extraction.

After uploading release assets, paste their browser-download URLs into the notebook:

```python
DATA_ZIP_URL = "https://github.com/tomjiu/isekaijoucho_colab_repo/releases/download/v0-data/isekaijoucho_source_audio.zip"
CHECKPOINT_ZIP_URL = "https://github.com/tomjiu/isekaijoucho_colab_repo/releases/download/v0-data/isekaijoucho_epoch1_checkpoint.zip"
PREPROCESSED_ZIP_URL = "https://github.com/tomjiu/isekaijoucho_colab_repo/releases/download/v0-data/isekaijoucho_preprocessed_logs.zip"
```

`CHECKPOINT_ZIP_URL` can be left empty to train from pretrained HiFi-GAN instead of resuming local epoch 1.
`PREPROCESSED_ZIP_URL` should stay filled if you want Colab to skip preprocessing and start training faster.

## Automatic Backups

The notebook includes an optional GitHub backup token prompt. For automatic backups, create a fine-grained GitHub token for this repository only with:

- Repository access: `tomjiu/isekaijoucho_colab_repo`
- Permissions: `Contents` -> `Read and write`

Paste it into the notebook's `GitHub backup token` cell. During training, each exported epoch model such as `isekaijoucho_whispertrim_2e_2010s.pth` is uploaded to:

`https://github.com/tomjiu/isekaijoucho_colab_repo/releases/tag/colab-model-backups`

Leave the token empty if you only want the final Colab download.

## Expected Speed

On a Colab T4, `batch_size=6` or `8` is usually much faster than the local GTX 1650 `batch_size=1`. If Colab reports CUDA OOM, lower `BATCH_SIZE` in the notebook.
