from sentence_transformers import SentenceTransformer, util,LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import csv
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
# model_save_path = 'all-MiniLM-L6-v2'
# model_save_path = 'sentence-transformers/sentence-t5-xl'
model_save_path = 'all-mpnet-base-v2'
test_dataset_path = "../data/stsb.csv" 

model = SentenceTransformer(model_save_path)

relation_samples = []
structure_samples = []

with open(test_dataset_path, "r", encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:

        score = float(row['relationscore']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['relation']], label=score)
        relation_samples.append(inp_example)

        score = float(row['structurescore']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['structure']], label=score)
        structure_samples.append(inp_example)

# Evaluate the model on STSbenchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(relation_samples, name='relation')
test_evaluator(model)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(structure_samples, name='structure')
test_evaluator(model)
