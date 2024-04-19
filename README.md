<div align="center">

# [FG 2024] Hyp-OC : Hyperbolic One Class Classification for Face Anti-Spoofing

[Kartik Narayan](https://kartik-3004.github.io/portfolio/) &emsp; [Vishal M. Patel](https://engineering.jhu.edu/faculty/vishal-patel/)  

Johns Hopkins University

<a href=''><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href=''><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

Official implementation of **[Hyp-OC : Hyperbolic One Class Classification for Face Anti-Spoofing]()**.
<hr />

## Highlights

Hyp-OC, is the first work exploring hyperbolic embeddings for one-class face anti-spoofing (OC-FAS).

1️⃣ We show that using hyperbolic space helps learn a better decision boundary than the Euclidean counterpart, boosting one-class face anti-spoofing performance.<br>
2️⃣ We propose Hyperbolic Pairwise Confusion Loss (Hyp-PC), that operates on the hyperbolic space inducing confusion within the hyperbolic feature space,
effectively stripping away identity information. Such disruption of features helps to learn better feature representations for the FAS task.<br>
3️⃣ We provide a hyperbolic formation of cross-entropy loss (Hyp-CE), that uses hyperbolic softmax logits and penalizes the network for every misclassification.<br>

<p align="center" width="100%">
  <img src='assets/visual_abstract.png' height="75%" width="75%">
</p>

> **<p align="justify"> Abstract:** *Face recognition technology has become an integral part of modern security systems
> and user authentication processes. However, these systems are vulnerable to spoofing attacks and can easily be circumvented.
> Most prior research in face anti-spoofing (FAS) approaches it as a two-class classification task where models are trained
> on real samples and known spoof attacks and tested for detection performance on unknown spoof attacks. However, in practice,
> FAS should be treated as a one-class classification task where, while training, one cannot assume any knowledge regarding
> the spoof samples a priori. In this paper, we reformulate the face anti-spoofing task from a one-class perspective and
> propose a novel hyperbolic one-class classification framework. To train our network, we use a pseudo-negative class sampled
> from the Gaussian distribution with a weighted running mean and propose two novel loss functions: (1) Hyp-PC: Hyperbolic
> Pairwise Confusion loss, and (2) Hyp-CE: Hyperbolic Cross Entropy loss, which operate in the hyperbolic space. Additionally,
> we employ Euclidean feature clipping and gradient clipping to stabilize the training in the hyperbolic space. To the best of
> our knowledge, this is the first work extending hyperbolic embeddings for face anti-spoofing in a one-class manner. With
> extensive experiments on five benchmark datasets: Rose-Youtu, MSU-MFSD, CASIA-MFSD, Idiap Replay-Attack, and OULU-NPU, we
> demonstrate that our method significantly outperforms the state-of-the-art, achieving better spoof detection performance.* </p>

# :rocket: News
- [04/20/2024] 🔥 We release Hyp-OC.

# Framework
<p align="center" width="100%">
  <img src='assets/framework.png'>
</p>
Overview of the proposed pipeline: Hyp-OC. The encoder extracts facial features which are used to estimate the mean of Gaussian 
distribution utilized to sample pseudo-negative points. The real features and pseudo-negative features are then concatenated 
and passed to FCNN for dimensionality reduction. The low-dimension features are mapped to Poincaré Ball using *exponential map*. 
The training objective is to minimize the summation of the proposed loss functions Hyp-PC} and Hyp-CE. The result is a separating 
*gyroplane* beneficial for one-class face anti-spoofing.


# Installation
```bash
conda env create --file environment.yml
conda activate hypoc
```

# Download Models
The models can be downloaded manually from [HuggingFace](https://huggingface.co/kartiknarayan/facexformer) or using python:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir="./")
```
The directory structure should finally be:

```
  . ── facexformer ──┌── ckpts/model.pt
                     ├── network
                     └── inference.py                    
```
# Usage

Download trained model from [HuggingFace](https://huggingface.co/kartiknarayan/facexformer) and ensure the directory structure is correct.<br>
For demo purposes, we have released the code for inference on a single image.<br>
It supports a variety of tasks which can be prompted by changing the "task" argument. 

```python
python inference.py --model_path ckpts/model.pt \
                    --image_path image.png \
                    --results_path results \
                    --task parsing \
                    --gpu_num 0

-- task = [parsing, landmarks, headpose, attributes, age_gender_race, visibility]
```
The trained models are stored in the specified "save_root".<br>
The training logs can be seen at "log_root"

## Citation
If you find Hyp-OC useful for your research, please consider citing us:

```bibtex
Coming Soon ...
```
## Contact
If you have any questions, please create an issue on this repository or contact at knaraya4@jhu.edu
