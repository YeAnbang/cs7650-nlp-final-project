# Continuous Sentimental-analysis-using-BERT


### Prerequisites

- Intermediate-level knowledge of Python 3 (NumPy and Pandas preferably, but not required)
- Exposure to PyTorch usage
- Basic understanding of Deep Learning and Language Models (BERT specifically)


### Project Outline

We use BERT as a encoder to generate a vector for each sentence. We also get a set of emotion bases with the same dimensions as our sentence encoding as input. A header that is similar to that in contrastive learning is applied to seperate sentence encoding from emotion base that is far away from GT emotion base in emotion embedding space, and makes sentence encoding and GT emotion bases closer. 

## Introduction

### What is BERT

BERT is a large-scale transformer-based Language Model that can be finetuned for a variety of tasks.

For more information, the original paper can be found [here](https://arxiv.org/abs/1810.04805). 

[HuggingFace documentation](https://huggingface.co/transformers/model_doc/bert.html)

[Bert documentation](https://characters.fandom.com/wiki/Bert_(Sesame_Street)) ;)


![Bert architecture](https://github.com/cipheraxat/Sentimental-analysis-using-BERT/blob/master/Images/BERT_diagrams.png)

## Exploratory Data Analysis and Preprocessing

I used the SMILE Twitter dataset.

_Wang, Bo; Tsakalidis, Adam; Liakata, Maria; Zubiaga, Arkaitz; Procter, Rob; Jensen, Eric (2016): SMILE Twitter Emotion dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.3187909.v2_


## Defining our Performance Metrics

Accuracy metric approach originally used in accuracy function in [this tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/#41-bertforsequenceclassification).

## Creating our Training Loop

Approach adapted from an older version of HuggingFace's `run_glue.py` script. Accessible [here](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128).

