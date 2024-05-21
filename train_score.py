# use conda environment m3e
import os
import warnings
import pandas as pd
from uniem.finetuner import FineTuner
from uniem.data_structures import RecordType, PairRecord, TripletRecord, ScoredPairRecord
import jsonlines

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    dataset = []
    with jsonlines.open('/home/majie/kongdechen/glm/AeroQA/aero_sentence_kg.jsonl') as f:
        # dataset = [obj for obj in f]
        for line in f:
            dataset.append({"sentence1":line["sentence1"], "sentence2":str(line["sentence2"]), "label":line["label"]})
    finetuner = FineTuner.from_pretrained('moka-ai/m3e-base', dataset=dataset)
    fintuned_model = finetuner.run(batch_size=64, epochs=9, output_dir='../en-finetuned-model-riddle')