import csv
import torch
from torch.nn import functional as F
import openai
import time
from scipy.stats import spearmanr
import os
import pinecone

openai.api_key = "sk-xxx"

# # find your PINECONE_API_KEY
# PINECONE_API_KEY = 'xxx'
# # find your PINECONE_ENVIRONMENT
# PINECONE_ENV = 'xxx'

# pinecone.init(
# api_key=PINECONE_API_KEY,
# environment=PINECONE_ENV
# )

# index_name = 'openai'
# index = pinecone.GRPCIndex(index_name)


# Define function to compute similarity between two sentences
def compute_similarity(sentence1, sentence2):
    response1 = openai.Embedding.create(input=sentence1, model="text-embedding-ada-002")
    response2 = openai.Embedding.create(input=sentence2, model="text-embedding-ada-002")
    embeddings1 = response1['data'][0]['embedding']
    embeddings2 = response2['data'][0]['embedding']
    tensor1 = torch.tensor(embeddings1).reshape(1, -1)
    tensor2 = torch.tensor(embeddings2).reshape(1, -1)
    similarity = F.cosine_similarity(tensor1, tensor2)
    return similarity.item(), embeddings1, embeddings2

# def upsert_pinecone(i, sentence1, emb0, emb1, emb2):
#     ids = []
#     xc = []
#     ids = [dataname+str(i)+'A',dataname+str(i)+'B',dataname+str(i)+'C']
#     metadata = [{'text':item} for item in [sentence1[0],sentence1[1],sentence1[2]]]
#     xc = [emb0,emb1,emb2]
#     records = zip(ids, xc, metadata)
#     return records

# Load CSV file with sentences and labels
dataname = 'cqa'
filename = dataname+'.csv'
sentences = []
sentences2 = []
labels = []
labels2 = []
with open(filename, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        sentence1 = row[0]
        sentence2 = row[1]
        sentence3 = row[3]
        label = row[2]
        label2 = row[4]
        sentences.append((sentence1, sentence2, sentence3))
        labels.append(float(label))
        labels2.append(float(label2))
print("Read test data successfully!")

# Compute similarity scores for all pairs of sentences
similarities = []
sims2 = []

for i, sentence1 in enumerate(sentences):
        time.sleep(2)
        print("No.",i,sentence1[0],sentence1[1], sentence1[2])
        similarity, emb0, emb1 = compute_similarity(sentence1[0], sentence1[1])
        sim2, emb0, emb2 = compute_similarity(sentence1[0],sentence1[2])
        # records = upsert_pinecone(i = i, sentence1 = sentence1, emb0=emb0,emb1=emb1,emb2=emb2)
        # index.upsert(vectors=records)
        similarities.append(similarity)
        sims2.append(sim2)

# Calculate Spearman correlation between similarities and labels
correlation, pvalue = spearmanr(similarities, labels)
print(f'Relation Spearman correlation: {correlation:.4f}, p-value: {pvalue:.3f}')
correlation2, pvalue = spearmanr(sims2, labels2)
print(f'Structure Spearman correlation: {correlation2:.4f}, p-value: {pvalue:.3f}')