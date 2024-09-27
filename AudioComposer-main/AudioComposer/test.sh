
CUDA_VISIBLE_DEVICES=0 python test.py --scale 5  --vocoder-ckpt  your_path/bigvnat \
    -b logs/train_config.yaml  --outdir your_path_for_output -r "your_path_weights/audio_composer.ckpt" \
    --test_file your_path_dataset/test.json