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
        prompt_e = open("data/prompt/example/a/10.txt", "r").read()
    else:
        prompt_e = open("data/prompt/example/b/6.txt", "r").read()
    
    #prompt_list = []
    #for i in range(1,13):
    prompt = open("data/prompt/b/8.txt", "r").read()
    #prompt_list.append(prompts_e)

    length = len(file)
    bar = tqdm.trange(length)
    mode_setting = ["no_context", "neg_context", "pos_context"]
    config_setting = ["pos", "neg", "mix"]
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

        for mode in mode_setting:
            #print("------------------ mode: ",  mode)
            if right == "wrong" and mode == "no_context":
                continue
            if right == "wrong" and mode == "neg_context":
                continue
            if right == "right" and mode == "pos_context":
                continue

            for config in config_setting:
                #print("------------------ config: ",  config)
                example_list = []
                for i in range(1,7):
                    example = open("data/prompt/example/" + mode + "/" + name + "/e_" + str(i) + ".txt", "r").read()
                    example_list.append(example)
                example_test_list = []
                if config == "pos":
                    example_test_list = example_list[:3]
                if config == "neg":
                    example_test_list = example_list[3:]
                if config == "mix":
                    for i in range(3):
                        random_int = random.randint(0, 1)
                        #print(i + random_int*3)
                        example_test_list.append(example_list[i + random_int*3])
        
                prompt_example = copy.deepcopy(prompt_e)
                for i in range(1, 4):
                    token_string = "[example" + str(i) + "]"
                    if token_string not in prompt_example:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Keyerror!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        # print(token_string)
                        # print(prompt_example)
                        # break
                    prompt_example = prompt_example.replace(token_string, example_test_list[i-1])

                prompt_temple = prompt  + "\n" + prompt_example
                if mode == "no_context":
                    print("666")
                    question_temple = "Question: " + question + choice + "\n"
                else:
                    question_temple = "Context: " + context + "\n" + "Question: " + question + choice + "\n"
                #question_temple = "Context: " + context + "\n" + "Question: " + question + choice + "\n"
                #print(prompt_temple)
                #print(question_temple)
                #break
                response = llm(prompt_temple, question_temple)
                #print(question_temple)
                sample = copy.deepcopy(data)
                sample["response"] = response
                #sample["prompt_index"] = index % 12 + 1
                sample["mode"] = mode
                sample["config"] = config
                #break

                with jsonlines.open("result/" + name + "/" + "chatgpt/" + name + "_" + right +  \
                        "_" + args.prompt + "_example_fixed.jsonl", 'a') as fw:
                        fw.write(sample)




    