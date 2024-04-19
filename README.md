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
