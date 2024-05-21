# use conda environment glm
from sentence_transformers import SentenceTransformer, util
import os
import jsonlines
import random
import torch
import torch.nn.functional as F
import sys
import jsonlines
from nebula.NebulaDatabase import nebula_database

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

model = SentenceTransformer("finetuned-model-riddle/model")
model = model.to(device)


def process_question_and_triples(question, triples):
    for i, t in enumerate(triples):
        triples[i] = str(t)
        sentences = [question]
        sentences.extend(triples)
    return sentences


def prompt_template(question, triples, scores=None, is_sort=True):
    if scores == None:
        triples_str = ",\n".join([str(triple) for triple in triples])
        prompt = f"""根据这些三元组，回答问题。三元组：{triples_str}\n 问题：{question}\n 答案："""
        return prompt
    
    # print(triples, type(triples))
    prefix = "根据提供的三元组，参考它们与问题的相关性回答问题"
    combined_triples = [eval(triple) + [f"与问题相关性{score:.4f}"] for triple, score in zip(triples, scores)]
    sorted_triples = sorted(combined_triples, key=lambda x: float(x[3][6:])) if is_sort else combined_triples
    triples_str = "[" + ",\n    ".join([str(triple[:]) for triple in sorted_triples]) + "]"
    prompt = f"""{prefix}
    三元组：{triples_str}
    根据这些三元组，参考其与问题的相关性，回答问题：
    问题：{question}
    答案："""
    return prompt


def format_result(nebula_result):
    """
    nebula_result: list of triples
    element of nebula_result is not string
    """
    for i, tri in enumerate(nebula_result):
        nebula_result[i] = eval(str(tri).replace("'", ""))
    return nebula_result


def get_scores(model, question, triples):
    """
    score triples with model
    """
    sentences = process_question_and_triples(question, triples)
    embedding = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(embedding, embedding)[0, 1:]
    return scores

def write_jsonlines(data, dst_path):
    with jsonlines.open(dst_path, mode="a") as f:
        f.write(data)

if __name__ == "__main__":
    split = "test"
    prompt_mode = ["with", "with"]
    with_sort = False
    with_score = False
    data_path = f"./Dataset/{split}/{split}_qa.jsonl"
    dst_path = f"./Dataset/{split}/{split}_{prompt_mode[with_sort]}_sort_{prompt_mode[with_score]}_score.jsonl"
    nebula_db = nebula_database()
    with jsonlines.open(data_path) as f:
        data = [obj for obj in f]
    qa_pair = {}
    for line in data:
        entity = line["实体"]
        question = line["问题"]
        answer = line["答案"]
        triples = []
        for e in entity:
            results = nebula_db.search(entity_name=e)
            triples.extend(format_result(results))
        scores = get_scores(model, question, triples)
        prompt = prompt_template(question, triples, scores, is_sort=True)
        qa_pair = {"input": prompt, "answer": f"{answer}"}
        write_jsonlines(qa_pair, dst_path)
    nebula_db.session.release()
    if split == "train":
        os.rename(dst_path, dst_path.replace("jsonl", "json"))