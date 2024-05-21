# Mecha-QA

This repository contains the code for the ***Relevance Prompts Based Knowledge Graph Question Answering*** method and the ***Mecha-QA*** dataset.

## Dataset

The Mecha-QA dataset is a knowledge graph question answering dataset focused on mechanical manufacturing. It comprises two sub-domains: traditional mechanical manufacturing and additive manufacturing. A knowledge graph is provided for each sub-domain. Based on these knowledge graphs, we used LLM to construct their corresponding question-answer pairs. Each pair includes the question, answer, entities, and the corresponding triples, totaling 1083 question-answer pairs.

More details [here](./dataset)

## Usage

### Getting Started

```bash
pip install -r requirements.txt
```

### Train Scoring Model

We use [m3e](https://huggingface.co/moka-ai/m3e-base) model as our scoring model. To finetune this model, please first process the question and serialized triples into JSON lines with the following keys: sentence1, sentence2, and label. A label of 0 denotes irrelevant, while a label of 1 denotes relevant.

Run following command:

```bash
python train_score.py
```

### Train LLM

#### Pre-process Data

After training the scoring model, pre-process the dataset before training the LLM. Run the following commands respectively:

```bash
python make_qa4ft.py
python make_qa4ft_aero.py
python make_qa4ft_3d.py
```

#### PEFT Finetuing

In our work, we use the **ChatGLM-6B** model and the **p-tuning v2** method to finetune it. For more details, please refer to [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning).

#### Evaluate

1. set `ptuning_checkpoint`  and `test_raw_file` in [evaluate.py](./evaluate.py)

2. ```bash
   python evaluate.py
   ```
