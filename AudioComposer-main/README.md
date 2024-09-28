# AudioComposer: Towards Fine-grained Audio Generation with Natural Language Descriptions
This repo contains our official implementation of <strong> AudioComposer </strong>. For the generated audio, Please refer to [[Demo]](https://lavendery.github.io/AudioComposer/). You can find our paper from [[Paper]](https://arxiv.org/abs/2409.12560).

## TODOs
- [x] Release paper and demo page.
- [x] Release pretrained weights.
- [x] Release inference code.
- [ ] Release training code.

## Pretrained Models
Models can be downloaded [here](https://huggingface.co/lavendery/AudioComposer/tree/main).
```
wget https://huggingface.co/lavendery/AudioComposer/resolve/main/audio_composer.ckpt?download=true
```

## Installation
```
conda create -n audiocomposer python=3.9
conda activate audiocomposer

git clone https://github.com/lavendery/AudioComposer.git
cd AudioComposer-main/AudioComposer
pip install -r requirements.txt

# infer
bash test.sh
```

## Acknowledgments
We would like to express our gratitude to several excellent repositories for making their code available to the public.
* [Make-An-Audio-2](https://github.com/bytedance/Make-An-Audio-2)
* [Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio)
* [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X)

## Citation
```bibtex
@misc{wang2024audiocomposerfinegrainedaudiogeneration,
      title={AudioComposer: Towards Fine-grained Audio Generation with Natural Language Descriptions}, 
      author={Yuanyuan Wang and Hangting Chen and Dongchao Yang and Zhiyong Wu and Helen Meng and Xixin Wu},
      year={2024},
      eprint={2409.12560},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2409.12560}, 
}
```