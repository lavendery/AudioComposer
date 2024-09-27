# AudioComposer: Towards Fine-grained Audio Generation with Natural Language Descriptions
This repo contains our official implementation of <strong> AudioComposer </strong>. 
[[Demo]](https://lavendery.github.io/AudioComposer/) [[Code]](https://github.com/lavendery/AudioComposer/tree/main/AudioComposer-main)

## TODOs
- [x] Release paper and demo page.
- [x] Release pretrained weights.
- [x] Release inference code.
- [ ] Release training code.

## Pretrained Models
Models can be downloaded [here](https://huggingface.co/lavendery/AudioComposer/tree/main).

## Installation
```
conda create -n audiocomposer python=3.9
conda activate audiocomposer
cd AudioComposer
pip install -r requirements.txt

# infer
bash test.sh
```


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