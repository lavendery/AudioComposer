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

## Data Format

### For Train
For the train.json and val.json, the format should be as follows:
```
# AudioCondition: 
{"dataset": "strong", "location": "data/AudioCondition/train/Y5tlDjxIa6i0.wav", "captions": "A rail transport makes a tapping sound, followed by an air brake.", "data_numpy": [["Rail transport", "Tap", "Air brake", "Generic impact sounds"], ["0.109", "9.606", "9.077", "6.653"], ["10.0", "9.722", "10.0", "6.741"]], "pitch_category": null, "energy_category": null}
...
...
# AudioTPE:
{"dataset": "FSD50K", "location": "data/AudioTPE/FSD50K/FSD50K.dev_audio/113131.wav", "captions": "Knock,Door,Domestic_sounds_and_home_sounds", "data_numpy": [[""], [""], [""]], "pitch_category": "normal pitch", "energy_category": "normal energy"}
...
...
# AudioCaps:
{"dataset": "audiocaps", "location": "data/audiocaps/train/Y6e8PzgsmKL8.wav", "captions": "A cat meows then something breaks", "data_numpy": [["A cat meows then something breaks"], ["0.0"], ["9.411906"]], "pitch_category": null, "energy_category": null}
...
...
```

### For Test
For the AudioCondition test set, the test.json should be as follows (The wav path is just GroundTruth, it can be none during inference. Only captions, start_time, and end_time are required.):
```
{"dataset": "strong", "location": "data/AudioCondition/test/Y_iL8G7GTNo0.wav", "captions": ["Water tap, faucet", "Female speech, woman speaking"], "start_time": [502, 740], "end_time": [732, 1000]}
```
For the AudioTPE test set, the test.json should be as follows (These wav paths are just GroundTruth, they can be none during inference. Only captions, start_time, and end_time are required.):
```
{"dataset": "AudioTPE", "location": "data/AudioTPE/ref/mix_61.wav", "select_file": ["data/FSD50K/FSD50K.dev_audio/101309.wav", "data/FSD50K/FSD50K.dev_audio/67475.wav"], "captions": ["Cheering,Shout,Human_group_actions,Human_voice", "Explosion"], "start_time": [0, 843], "end_time": [843, 1000], "pitch_category": ["normal pitch", "normal pitch"], "energy_category": ["normal energy", "normal energy"]}
```
For the AudioCaps test, the test.json should be as follows (The wav path is just GroundTruth, it can be none during inference. Only captions, start_time, and end_time are required.):
```
{"dataset": "audiocaps", "location": "data/audiocaps_test/Y6TO9PEGpZcQ.wav", "captions": ["An emergency siren wailing followed by a large truck engine running idle"], "start_time": [0], "end_time": [1000]}
```
After these test.json files are automatically processed by the code, the final natural language input should be as follows:
```
Water tap, faucet, Start at 5.02s and End at 7.32s. Female speech, woman speaking, Start at 7.4s and End at 10s.
```

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
