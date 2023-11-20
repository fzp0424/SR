from langchain.chat_models import ChatOpenAI
import csv
import os
from dotenv import load_dotenv, find_dotenv
import openai
import pandas as pd
from action.write import write_sent
from action.write_check import write_check
from action.score import sts_score

# add your openai key in .env as OPENAI_API_KEY
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
temperature = 0.7
llm = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=temperature)  

file_path='example.csv'
raw = pd.read_csv(file_path, encoding='utf-8',header = None)
raw = raw.columns[0]

rawfile = "rawdata.csv"
if not os.path.isfile(rawfile):
    df = pd.DataFrame(columns=['sentence1', 'sentence2', 'check', 'score'])  
    df.to_csv(rawfile, index=False)  
    print("Successfully create raw data file!")

batch_size = 10 # If you change the batch_size, you need to change the template in action accordingly
for chunk in pd.read_csv(file_path, encoding='utf-8', header=None, chunksize=batch_size):
    df = pd.read_csv(rawfile)  
    sentences = chunk.iloc[:, 0].tolist()  # Extract sentences from the chunk
    is_in_area = df['sentence1'].isin([sentences[0]]).any() 
    if is_in_area: 
        continue
    else:
        df = pd.read_csv(rawfile)  
        # llm = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=temperature)  
        write_dict = write_sent(llm = llm, raw_data= sentences)
        check_dict = write_check(llm = llm, ans1_dict = write_dict)
        sts_score_dict = sts_score(llm = llm, ans1_dict = write_dict)

        with open(rawfile, "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            for i in range(1, len(write_dict) // 2 + 1):
                tag = str(i)+'.'
                tag_pair = 'Change'+str(i)+':'
                tag_check = 'Check'+str(i)+':'
                tag_score = tag + ' Scoring'
                write_temp = [check_dict[tag], check_dict[tag_pair], check_dict[tag_check][:1], sts_score_dict[tag_score]]
                csv_writer.writerow(write_temp)  # Write each item in a separate row