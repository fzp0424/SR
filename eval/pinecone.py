import os
import time
import pinecone
import csv
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr


# find your PINECONE_API_KEY
PINECONE_API_KEY = 'xxx'
# find your PINECONE_ENVIRONMENT
PINECONE_ENV = 'xxx'

pinecone.init(
api_key=PINECONE_API_KEY,
environment=PINECONE_ENV
)

index_name = 'openai'
# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536, #openai ada
        metric='cosine'
    )
    # wait a moment for the index to be fully initialized
    time.sleep(1)

# now connect to the index
index = pinecone.GRPCIndex(index_name)

#calculate similarity between two embeddings
def compute_similarity(embeddings1, embeddings2):
    tensor1 = torch.tensor(embeddings1).reshape(1, -1)
    tensor2 = torch.tensor(embeddings2).reshape(1, -1)
    similarity = F.cosine_similarity(tensor1, tensor2)
    return similarity.item()


# Load CSV file with sentences and labels
dataname = 'twitter'
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
        print("No.",i,sentence1[0],sentence1[1], sentence1[2])
        id0 = dataname+str(i)+'A'
        id1 = dataname+str(i)+'B'
        id2 = dataname+str(i)+'C'
        ids = [id0, id1, id2]
        pp = index.fetch(ids)['vectors']
        emb0 = pp[id0]['values']
        emb1 = pp[id1]['values']
        emb2 = pp[id2]['values']
        similarity = compute_similarity(emb0, emb1)
        sim2 = compute_similarity(emb0, emb2)
        similarities.append(similarity)
        sims2.append(sim2)

# Calculate Spearman correlation between similarities and labels
correlation, pvalue = spearmanr(similarities, labels)
print(f'Relation Spearman correlation: {correlation:.4f}, p-value: {pvalue:.3f}')
correlation2, pvalue = spearmanr(sims2, labels2)
print(f'Structure Spearman correlation: {correlation2:.4f}, p-value: {pvalue:.3f}')