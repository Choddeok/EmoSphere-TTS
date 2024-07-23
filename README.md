# EmoSphere-TTS: Emotional Style and Intensity Modeling via Spherical Emotion Vector for Controllable Emotional Text-to-Speech <br><sub>The official implementation of EmoSphere-TTS</sub>
##  <a src="https://img.shields.io/badge/cs.CV-2406.07803-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2406.07803"> <img src="https://img.shields.io/badge/cs.CV-2406.07803-b31b1b?logo=arxiv&logoColor=red"></a>|[Demo page](https://emosphere-tts.github.io/)

**Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Sang-Hoon Lee, Seong-Whan Lee**

Department of Artificial Intelligence, Korea University, Seoul, Korea  

## Abstract
Despite rapid advances in the field of emotional text-to-speech (TTS), recent studies primarily focus on mimicking the average style of a particular emotion. As a result, the ability to manipulate speech emotion remains constrained to several predefined labels, compromising the ability to reflect the nuanced variations of emotion. In this paper, we propose EmoSphere-TTS, which synthesizes expressive emotional speech by using a spherical emotion vector to control the emotional style and intensity of the synthetic speech. Without any human annotation, we use the arousal, valence, and dominance pseudo-labels to model the complex nature of emotion via a Cartesian-spherical transformation. Furthermore, we propose a dual conditional adversarial network to improve the quality of generated speech by reflecting the multi-aspect characteristics. The experimental results demonstrate the model’s ability to control emotional style and intensity with high-quality expressive speech.

![240312_model_overview_1](https://github.com/Choddeok/EmoSphere-TTS/assets/77186350/913610da-bfcc-4e60-b8fe-c1172b8dc154)

------
## Training Procedure

### Environments
```
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
bash mfa_usr/install_mfa.sh # install force alignment tools
```

### 1. Preprocess data

- We use ESD database, which is an emotional speech database that can be downloaded here: https://hltsingapore.github.io/ESD/. 

```bash
sh preprocessing.sh
```

### 2. Training TTS module and Inference  
```bash
sh train_run.sh
```

### 3. Pretrained checkpoints
- TTS module trained on 160k [[Download]](https://works.do/5eA33VN)

## Acknowledgements
**Our codes are based on the following repos:**
* [NATSpeech](https://github.com/NATSpeech/NATSpeech)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [BigVGAN](https://github.com/NVIDIA/BigVGAN)
