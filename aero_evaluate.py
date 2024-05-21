# conda activate glm
# 修改ptuning-checkpoint, test_file, dst_file;如果不用已处理的数据做测试，就用main里边最下面注释的代码，注意修改数据格式
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import torch
# from nebula.NebulaDatabase import nebula_database
from sentence_transformers import SentenceTransformer,util
import jsonlines
import random
import sys
import csv
import re
import sqlite3


sys.path.append("./THUDM/chatglm-6b")

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

scoring_model = SentenceTransformer('en-finetuned-model-riddle/model')
scoring_model = scoring_model.to('cuda')

model_name_or_path = "THUDM/chatglm-6b"
ptuning_checkpoint = "/home/majie/kongdechen/glm/ChatGLM-6B/ptuning/output/aero-qa-chatglm-6b-pt-128-5e-4-do_with_sort_with_score/checkpoint-2000"
# ptuning_checkpoint = ""
random_score = False
fixed_score = False
is_sort = True

pre_seq_len = 128
prefix_projection = False

test_raw_file = r"AeroQA/1hop/qa_test.csv"
# test_file = r"Dataset/test/test_4glm_with_sort_wo_score.jsonl"
# dst_file = r"test_results_with_sort_wo_score_0908.jsonl"
test_file = r"AeroQA/1hop/test/test_4glm_fs_with_sort_with_score.jsonl"
dst_file = r"aero_glm_fs_with_sort_with_score_0921.jsonl"
db_pth = r"./AeroQA/aviation.db"



def process_question_and_triples(question, triples):
    for i, t in enumerate(triples):
        triples[i] = str(t[:])
        sentences = [question]
        sentences.extend(triples)
    return sentences


def get_scores(model, question, triples, random_score=False, fixed_score=False):
    """
    score triples with model
    """
    sentences = process_question_and_triples(question, triples)
    if random_score:
        return [random.random() for _ in range(len(sentences)-1)]
    if fixed_score:
        return [0.800 for _ in range(len(sentences)-1)]
    embedding = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(embedding, embedding)[0, 1:]
    return scores


def load_model(model_name_or_path, ptuning_checkpoint, pre_seq_len=128, prefix_projection=False):
    """
    load glm model with p-tuning ckpt 
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = pre_seq_len
    config.prefix_projection = prefix_projection
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True).half().cuda()
    if ptuning_checkpoint == "":
        return model, tokenizer
    prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.eval()
    return model, tokenizer


def prompt_template(question, triples, scores=None, is_sort=True):
    if scores == None:
        triples_str = ",\n".join([str(triple) for triple in triples])
        prompt = f"""triples：{triples_str}
        Based on these triples, refer to their relevance to the question, and answer the question：
        question：{question}
        answer："""
        return prompt
    
    # print(triples, type(triples))
    combined_triples = [eval(triple) + [f"relevance to the question: {score:.4f}"] for triple, score in zip(triples, scores)]
    sorted_triples = sorted(combined_triples, key=lambda x: float(x[3].split(": ")[-1])) if is_sort else combined_triples
    triples_str = "[" + ",\n    ".join([str(triple[:]) for triple in sorted_triples]) + "]"
    prompt = f"""triples：{triples_str}
    Based on these triples, refer to their relevance to the question, and answer the question：
    question：{question}
    answer："""
    # print(len(prompt))
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


def revise_prompt(prompt):
    if len(prompt) < 2000:
        return prompt
    temp = prompt.rsplit("triples", 1)
    temp2 = temp[-1].split("    question：")
    prompt = temp[0] + "triples" + temp2[0][:1000] + "    question：" + temp2[-1]
    return prompt



def generate_answer(question, entity, model, tokenizer, prompt=""):
    # scores = None
    if prompt == "":
        triples = []
        for e in entity:
            result = db_search.search(e)
            triples.extend(format_result(result))
        scores = get_scores(scoring_model, question, triples, random_score=random_score, fixed_score=fixed_score)
        prompt = prompt_template(question, triples, scores=scores, is_sort=is_sort)
    # if len(prompt) > 2000:
    #     prompt = prompt[-2000:]
    # print(len(prompt))
    # if len(prompt) > 2000:
    #     temp = prompt.split("Based on these triples")
    #     prompt = temp[0][:2000] + "Based on these triples" + temp[-1]
    prompt = revise_prompt(prompt)
    # print(prompt)
    response, _ = model.chat(tokenizer, prompt)
    return response


def write_jsonlines(data, dst_path):
    with jsonlines.open(dst_path, mode="a") as f:
        f.write(data)


if __name__ == "__main__":
    history = []
    # nebula_db = nebula_database()
    model, tokenizer = load_model(model_name_or_path, ptuning_checkpoint)
    with jsonlines.open(test_file) as f:
        data = [obj for obj in f]
    with open(test_raw_file) as f:
        raw_data = csv.reader(f)
        next(raw_data)
        for line,inputs in zip(raw_data, data):
            question = line[0].replace("[", "").replace("]", "")
            entity = re.findall(r'\[(.*?)\]', line[0])
            gt = line[-1]
            prompt = inputs["input"]
            response = generate_answer(question, entity, model, tokenizer, prompt )
            out = {"问题": question, "ground_truth": gt, "generated_answer": response}
            write_jsonlines(out, dst_file)
        # nebula_db.session.release()

