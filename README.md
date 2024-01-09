# SR Benchmark for Text Embedding

This repository contains the code and data for our paper [[How Well Do Text Embedding Models Understand Syntax?]](https://arxiv.org/abs/2311.07996) which has been accepted as the Findings of EMNLP 2023.

# Overview

We establish an evaluation set, named **SR**, to scrutinize the capability for syntax understanding of text embedding models from two crucial syntactic aspects: **S**tructural heuristics, and **R**elational understanding among concepts.

![fig2.svg](fig/fig2.svg)

# Data

Our **SR** benchmark contains source sentences from `STS-B`, `CQADupStack`, `Twitter`, `BIOSSES`, `SICK-R`, and `AskUbuntu`. 

# Environment

```python
pip install -r requirements.txt
```

# Evaluation

## Sentence Encoder

Take SentenceTransformer as an example,

```python
cd eval
python sbert_test.py
```

## OpenAI Ada

Set your own keys in `openai_eval.py`

## Stored embeddings

Take [pinecone](https://www.pinecone.io/) as an example, fill your index and API key in `pinecone.py`

# Generate

## Step1

Set your OpenAI keys in `.env`

## Step2

Check the prompt used in `/action` This project is built on [LangChain](https://github.com/langchain-ai/langchain), feel free to search for your own prompt/template to generate your sentences. Because of batch inference, you need to change the template and json parser if you change the `batch_size` in `collect.py`.

## Step3

Run a demo to see if you can run this project successfully. Then replace your file with `example.csv`

```python
python collect.py
```

# Citation
```
@inproceedings{zhang2023well,
  title={How Well Do Text Embedding Models Understand Syntax?},
  author={Zhang, Yan and Feng, Zhaopeng and Teng, Zhiyang and Liu, Zuozhu and Li, Haizhou},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={9717--9728},
  year={2023}
}
```

