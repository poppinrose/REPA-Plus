TARGET_PATH="./data/imagenet"

torchrun --nproc_per_node=8 preprocessing/preprocess.py \
    --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=${TARGET_PATH}/vae-sd \
    --dest-images=${TARGET_PATH}/images \
    --batch-size=32 \
    --resolution=512 \
    --transform=center-crop-dhariwal