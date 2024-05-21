# use conda environment glm_lora
from sentence_transformers import SentenceTransformer, util
import os
import jsonlines
import random
import torch
import torch.nn.functional as F
import sys
import jsonlines
import sqlite3

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


def prompt_template(question, triples, scores, is_sort, with_score):
    if scores == None:
        triples_str = ",\n".join([str(triple) for triple in triples])
        prompt = f"""根据这些三元组，回答问题。三元组：{triples_str}\n 问题：{question}\n 答案："""
        return prompt
    
    combined_triples = [eval(triple) + [f"与问题相关性{score:.4f}"] for triple, score in zip(triples, scores)]
    sorted_triples = sorted(combined_triples, key=lambda x: float(x[3][6:])) if is_sort else combined_triples
    if with_score:
        triples_str = "[" + ",\n    ".join([str(triple[:]) for triple in sorted_triples]) + "]"
        prompt = f"""三元组：{triples_str}\n根据这些三元组，参考其与问题的相关性，回答问题。\n问题：{question}\n答案："""

    else:
        triples_str = "[" + ",\n    ".join([str(triple[:3]) for triple in sorted_triples]) + "]"
        prompt = f"""三元组：{triples_str}\n根据这些三元组，回答问题。\n问题：{question}\n答案："""

    return prompt


def format_result(results):
    # [(),(),...] -> [[],[],...]
    for i,r in enumerate(results):
        results[i] = list(r)
    return results


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

def db_search(entity_name, database):
    # 打开数据库连接
    # conn = sqlite3.connect(database)
    cursor = database.cursor()

    # 执行查询
    cursor.execute('SELECT * FROM triples WHERE subject = ? or predicate = ? or object = ?', (entity_name, entity_name, entity_name))

    # 获取所有匹配的三元组
    matching_triples = cursor.fetchall() # [(),(),....]
    # conn.close()
    return matching_triples

def generate_dataset(triples, scores):

    for with_score in mode:
        for with_sort in mode:
            dst_path = f"./Dataset/{split}_3d/{split}_{prompt_mode[with_sort]}_sort_{prompt_mode[with_score]}_score_3d.jsonl"
            prompt = prompt_template(question, triples, scores, is_sort=with_sort, with_score=with_score)
            qa_pair = {"input": prompt, "answer": f"{answer}"}
            write_jsonlines(qa_pair, dst_path)

if __name__ == "__main__":
    split = "test"
    data_path = f"./Dataset/{split}_3d/kbqa_3d.jsonl"
    with jsonlines.open(data_path) as f:
        data = [obj for obj in f]
    
    prompt_mode = {True: "with", False: "wo"}
    mode = [True, False]
    
    db_path = r"./Dataset/test_3d/kg_3d.db"
    db = sqlite3.connect(db_path)

    qa_pair = {}
    for line in data:
        entity = line["实体"]
        question = line["问题"]
        answer = line["答案"]
        triples = []
        for e in entity:
            results = db_search(entity_name=e, database=db)
            triples.extend(format_result(results))
        scores = get_scores(model, question, triples)
        generate_dataset(triples, scores)
