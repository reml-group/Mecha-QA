# conda activate glm
# 修改ptuning-checkpoint, test_file, dst_file;如果不用已处理的数据做测试，就用main里边最下面注释的代码，注意修改数据格式
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import torch
from nebula.NebulaDatabase import nebula_database
from sentence_transformers import SentenceTransformer,util
import jsonlines
import random
import sys

sys.path.append("./THUDM/chatglm-6b")

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

scoring_model = SentenceTransformer('finetuned-model-riddle/model')
scoring_model = scoring_model.to('cuda')

model_name_or_path = "THUDM/chatglm-6b"
random_score = False
fixed_score = False
is_sort = True

pre_seq_len = 128
prefix_projection = False

test_raw_file = r"Dataset/test_3d/kbqa_3d.jsonl"

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
        prompt = f"""三元组：{triples_str}
        根据这些三元组，回答问题：
        问题：{question}
        答案："""
        return prompt
    
    # print(triples, type(triples))
    combined_triples = [eval(triple) + [f"与问题相关性{score:.4f}"] for triple, score in zip(triples, scores)]
    sorted_triples = sorted(combined_triples, key=lambda x: float(x[3][6:])) if is_sort else combined_triples
    triples_str = "[" + ",\n    ".join([str(triple) for triple in sorted_triples]) + "]"
    prompt = f"""三元组：{triples_str}
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
        try:
            nebula_result[i] = eval(str(tri).replace("'", ""))
        except:
            nebula_result[i] = eval(str(tri).replace("'", "").replace('"Cr”', "Cr"))
    return nebula_result


def generate_answer(question, entity, model, tokenizer, nebula_db, prompt=""):
    # scores = None
    if prompt == "":
        triples = []
        for e in entity:
            result = nebula_db.search(e)
            triples.extend(format_result(result))
        scores = get_scores(scoring_model, question, triples, random_score=random_score, fixed_score=fixed_score)
        prompt = prompt_template(question, triples, scores=scores, is_sort=is_sort)
    response, _ = model.chat(tokenizer, prompt)
    return response


def write_jsonlines(data, dst_path):
    with jsonlines.open(dst_path, mode="a") as f:
        f.write(data)


if __name__ == "__main__":
    history = []
    # nebula_db = nebula_database()
    test_file_list = []
    dst_file_list = []
    ptuning_checkpoint_list = []
    for with_score in ["with", "wo"]:
        for with_sort in ["with", "wo"]:
            test_file_list.append(f"Dataset/test_3d/test_{with_sort}_sort_{with_score}_score_3d.jsonl")
            dst_file_list.append(f"results/3d/test_{with_sort}_sort_{with_score}_score_3d.jsonl")
            ptuning_checkpoint_list.append(f"ChatGLM-6B/ptuning/output/qa-chatglm-6b-pt-128-5e-4-{with_sort}_sort_{with_score}_score/checkpoint-2000")

    nebula_db = None
    
    with jsonlines.open(test_raw_file) as f:
        raw_data = [obj for obj in f]
    
    for test_file, dst_file, ptuning_checkpoint in zip(test_file_list, dst_file_list, ptuning_checkpoint_list):

        
        model, tokenizer = load_model(model_name_or_path, ptuning_checkpoint)

        with jsonlines.open(test_file) as f:
            data = [obj for obj in f]
        for line,inputs in zip(raw_data, data):
            question = line["问题"]
            entity = line["实体"]
            gt = line["答案"]
            prompt = inputs["input"]
            # prompt = question
            response = generate_answer(question, entity, model, tokenizer, nebula_db, prompt )
            out = {"问题": question, "ground_truth": gt, "generated_answer": response, "triples": line["对应的三元组"]}
            write_jsonlines(out, dst_file)
        del model

