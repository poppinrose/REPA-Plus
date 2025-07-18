# REPA-Plus: a more conveniet codebase for REPA with some improvements

## Paper Using this codebase

ENA: Efficient N-dimensional Attention

## What's Changed

- Automated dataset download from huggingface, no manual download required.
- Parallel preprocessing.
- Train an ENA model alongside original SiT model.
- A cosine learning rate scheduler.
- Muon and AdamW as optimizers.

## Environment Setup

```bash
# Create and activate conda environment
conda create -n repa-plus python=3.12
conda activate repa-plus

pip install torch==2.6.0 torchvision==0.21.0 accelerate diffusers timm --use-pep517

pip install transformers datasets evaluate causal_conv1d einops scikit-learn wandb matplotlib deepspeed

# Install flash-attention, this is required if you like hybrid models
pip install flash-attn==2.7.4.post1

# A handy tool to monitor GPU
pip install nvitop

pip install -U "huggingface_hub[cli]" --use-pep517
pip install pillow==11.1.0 --use-pep517
pip install git+https://github.com/facebookresearch/pytorchvideo.git

pip install -e git+https://github.com/fla-org/fla-zoo.git
```

## Workflow

- First use `huggingface-cli login` and paste your token (visit huggingface website to get one) to login. Use `bash process.sh` for automated data download and preprocessing. Change the `nproc_per_node` to the number of GPUs you have. Change the `batch_size` accrording to your GPU memory. The default is set to 32, which should work for most GPUs with 40GB memory.
- Use `bash train.sh` to start training. You can tune the parameters in the script, such as `--warmup-steps`, `--batch-size`, etc.

## Huge Thanks

- REPA authors for their original work and codebase: [see here](https://github.com/sihyun-yu/REPA).
- Yuqian Hong for multi-GPU preprocessing scripts: [see this PR](https://github.com/sihyun-yu/REPA/pull/43).


## Plans (no time guarantee)

- T2I training.
- Some faster way to train the model. (open an issue or PR if you have any demonstrated ideas)
- Improving the codebase to be more convenient (open an issue or PR if you have any demonstrated ideas).