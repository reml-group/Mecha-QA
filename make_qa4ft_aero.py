# use conda environment glm
from sentence_transformers import SentenceTransformer, util
import os
import jsonlines
import random
import torch
import torch.nn.functional as F
import sys
import jsonlines
import sqlite3
import csv
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

model = SentenceTransformer("en-finetuned-model-riddle/model")
model = model.to(device)
db_pth = r"./AeroQA/aviation.db"


def process_question_and_triples(question, triples):
    for i, t in enumerate(triples):
        triples[i] = str(t)
        sentences = [question]
        sentences.extend(triples)
    return sentences


def prompt_template(question, triples, scores=None, is_sort=True):
    if scores == None:
        triples_str = ",\n".join([str(triple) for triple in triples])
        prompt = f"""triples：{triples_str}
        Based on these triples, and answer the question：
        question：{question}
        answer："""
        return prompt
    
    prefix = "Based on the given triples, refer to their relevance to the question, and answer the question."
    combined_triples = [eval(triple) + [f"relevance to the question: {score:.4f}"] for triple, score in zip(triples, scores)]
    sorted_triples = sorted(combined_triples, key=lambda x: float(x[3].split(": ")[-1])) if is_sort else combined_triples
    triples_str = "[" + ",\n    ".join([str(triple[:]) for triple in sorted_triples]) + "]"
    prompt = f"""{prefix}
    triples：{triples_str}
    question：{question}
    answer："""
    return prompt


def db_search(keyword, database=db_pth):
    # 打开数据库连接
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # 执行查询
    cursor.execute('SELECT * FROM triples WHERE subject = ? or predicate = ? or object = ?', (keyword, keyword, keyword))

    # 获取所有匹配的三元组
    matching_triples = cursor.fetchall() # [(),(),....]
    conn.close()
    return matching_triples


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


if __name__ == "__main__":
    # split = "dev"
    # split_type = ["train", "dev", "test"]
    split_type = ["test"]
    prompt_mode = ["wo", "with"]
    with_sort = True
    with_score = True
    for split in split_type:
        data_path = f"./AeroQA/1hop/qa_{split}.csv"
        dst_path = f"./AeroQA/1hop/{split}/{split}_4glm_{prompt_mode[with_sort]}_sort_{prompt_mode[with_score]}_score.jsonl"
        # nebula_db = nebula_database()
        with open(data_path, 'r') as file:
        # 使用csv.reader来读取文件内容
            csv_reader = csv.reader(file)
            next(csv_reader)
            qa_pair = {}
            for line in csv_reader:
                entity = re.findall(r'\[(.*?)\]', line[0])
                question = line[0].replace("[", "").replace("]", "")
                answer = line[-1]
                triples = []
                for e in entity:
                    results = db_search(keyword=e)
                    triples.extend(format_result(results))
                scores = get_scores(model, question, triples)
                # scores = None
                prompt = prompt_template(question, triples, scores, is_sort=True)
                qa_pair = {"input": prompt, "answer": f"{answer}"}
                write_jsonlines(qa_pair, dst_path)
        if not split == "test":
            os.rename(dst_path, dst_path.replace("jsonl", "json"))