# Generative AI-Powered Plugin for Robust Federated Learning in Heterogeneous IoT Networks

This is an official implementation of the following paper:
> Youngjoon Lee, Jinu Gong, and Joonhyuk Kang.
**[Generative AI-Powered Plugin for Robust Federated Learning in Heterogeneous IoT Networks](https://arxiv.org/abs/2410.23824)**  
_arXiv preprint arXiv:2410.23824_.

## Requirements
Please install the required packages as below

```pip install tensorboard scipy tqdm pandas torch ollama transformers```

## Federated Learning Techniques
This paper considers the following federated learning techniques
- FedAvg ([McMahan, Brendan, et al. AISTATS 2017](http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com))
- FedProx ([Li, Tian, et al. MLSys 2020](https://proceedings.mlsys.org/paper/2020/hash/38af86134b65d0f10fe33d30dd76442e-Abstract.html))
- FedRS ([Li, X. C., & Zhan, D. C. SIGKDD 2021](https://dl.acm.org/doi/10.1145/3447548.3467254))

## Edge AI Model
- google-bert/bert-base-uncased
- distilbert/distilbert-base-uncased
- dmis-lab/biobert-v1.1
- microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

## LLM
- Gemma 1.0:7B, Gemma 2.0:2B/9B
- Llama 3.1:8B, Llama 3.2:1B/3B
- Phi 2.0:3B, Phi 3.0:3B, Phi 3.5:3B

## Dataset
- Medical-Abstracts-TC-Corpus ([Schopf, T., Braun, D., & Matthes, F. NLPIR 2022](https://dl.acm.org/doi/abs/10.1145/3582768.3582795?casa_token=vmNfG1V1b8sAAAAA:oW1Kdt7H1adX5lHDrIEABSO942VNa7OBS1gw1eBZsLVrgbMyn_DZs7ZlO7hAk_XPnoL9S7ItZGg))
