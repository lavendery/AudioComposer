<p align="center">

  <h2 align="center"> AudioComposer: Towards Fine-grained Audio Generation with Natural Language Descriptions </h2>
  <p align="center">
        <a href="https://arxiv.org/abs/2409.12560">
        <img src='https://img.shields.io/badge/arXiv-red' alt='Paper Arxiv'></a> &nbsp; &nbsp;  &nbsp; 
        <a href='https://lavendery.github.io/AudioComposer/'>
        <img src='https://img.shields.io/badge/Project_Page-green' alt='Project Page'></a> &nbsp;&nbsp; &nbsp; 
        <a href="https://github.com/lavendery/AudioComposer/tree/main">
          <img src="https://img.shields.io/badge/Code-black?logo=github&logoColor=white" alt="Code">
        </a>&nbsp;&nbsp; &nbsp; 
  </p>
</p>

This repo contains our official implementation of <strong> AudioComposer </strong>. For the generated audio, Please refer to [[Demo]](https://lavendery.github.io/AudioComposer/). You can find our paper from [[Paper]](https://arxiv.org/abs/2409.12560).

## Abstract
Current Text-to-audio (TTA) models mainly use coarse text descriptions as inputs to generate audio, which hinders models from generating audio with fine-grained control of content and style. Some studies try to improve the granularity by incorporating additional frame-level conditions or control networks. However, this usually leads to complex system design and difficulties due to the requirement for reference frame-level conditions. To address these challenges, we propose AudioComposer, a novel TTA generation framework that relies solely on natural language descriptions (NLDs) to provide both content specification and style control information. To further enhance audio generative modeling, we employ flow-based diffusion transformers with the cross-attention mechanism to incorporate text descriptions effectively into audio generation processes, which can not only simultaneously consider the content and style information in the text inputs, but also accelerate generation compared to other architectures. Furthermore, we propose a novel and comprehensive automatic data simulation pipeline to construct data with fine-grained text descriptions, which significantly alleviates the problem of data scarcity in the area. Experiments demonstrate the effectiveness of our framework using solely NLDs as inputs for content specification and style control. The generation quality and controllability surpass stateof-the-art TTA models, even with a smaller model size. 

## TODOs
- [x] Release paper and demo page.
- [x] Release pretrained weights.
- [x] Release inference code.
- [x] Release training code.

## Pretrained Models
Models can be downloaded [here](https://huggingface.co/lavendery/AudioComposer/tree/main).
```
wget https://huggingface.co/lavendery/AudioComposer/resolve/main/audio_composer.ckpt?download=true
```

We use the same BIGVGAN vocoder weights as [Make-An-Audio-2](https://github.com/bytedance/Make-An-Audio-2?tab=readme-ov-file). You can download them and then put them into 'AudioComposer-main/AudioComposer/configs/model/bigvnat/'.

We also use the same Mel-VAE checkpoint as Make-An-Audio-2, You can download them from [here](https://www.modelscope.cn/datasets/lavendery/epoch/resolve/master/epoch%3D000032.ckpt) and put them into 'configs/model/vae/', please citation [Make-An-Audio-2](https://github.com/bytedance/Make-An-Audio-2?tab=readme-ov-file) if you use these checkpoints.

Additionally, you need to download [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) model and then put them into 'configs/model/google/flan-t5-large model'.

## Installation
```
git clone https://github.com/lavendery/AudioComposer.git
conda create -n audiocomposer python=3.9
conda activate audiocomposer

cd AudioComposer-main/AudioComposer
pip install -r requirements.txt

# Train
bash train.sh

# Inference
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
