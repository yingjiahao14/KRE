import os
import openai
import json
import sys
import random
import time
import torch 
import numpy as np
import argparse
import tqdm
import backoff 
import jsonlines
import time
import copy
os.chdir("../../")

openai.api_key = ""
	
@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError))
def llm(prompt, question):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages = [{"role": "system", "content": prompt},
                  {"role": "user", "content": question}],
      temperature = 0,
    )
    return response['choices'][0]['message']['content']

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--name', type=str)
parser.add_argument('--right', type=str)
parser.add_argument('--prompt', type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    print("====================", args.name, args.right)
    name = args.name
    right = args.right
    file = json.load(open("result_chatgpt/" + name + "/" + name + "_" + right + "_with_neg.json","r"))
    if args.prompt == "a":
        prompt = open("data/prompt/a/6.txt", "r").read()
    else:
        prompt = open("data/prompt/b/8.txt", "r").read()

    length = len(file)
    bar = tqdm.trange(length)
    answer_dict = {0: ' A: ', 1: ' B: ', 2: ' C: ', 3: " D: ", 4: ' E: ', 5: ' F: '}
    for idx, (data) in zip(bar,file):
        question = data["question"]
        answer = data["answer"]
        context = ""
        if right == "wrong":
            context = data[" golden_context"]
        elif right == "right":
            context = data["negative_context"]
        else:
            print("Error")
        choices = data["choices"]
        choice = ""
        for index, choi in enumerate(choices):
            choice += answer_dict[index] + choi
        
        prompt_temple = prompt 
        question_temple = "Context: " + context + "\n" + "Question: " + question + choice + "\n"
        #print(prompt_temple, question_temple)
        response = llm(prompt_temple, question_temple)

        sample = copy.deepcopy(data)
        sample["response"] = response

        with jsonlines.open("result/" + name + "/" + "chatgpt/" + name + "_" + right +  \
                            "_" + args.prompt + "_fixed_.jsonl", 'a') as fw:
            fw.write(sample)
        #break
        
        

    