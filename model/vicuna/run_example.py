# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import tqdm
import random
import argparse
from pathlib import Path
import copy

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

os.chdir("../../")


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    print("world----size:" + str(local_rank)  + str(world_size))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 1,
    max_seq_len: int = 4096,
    max_batch_size: int = 12,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )   
    prompt = open("data/prompt/a/" + "1" + ".txt", "r").read()
    prompt_a = open("data/prompt/example/a/" + "9" + ".txt", "r").read()
    prompt_b = open("data/prompt/example/b/" + "5" + ".txt", "r").read()
    prompt_list = {"a": prompt_a, "b":prompt_b}
    #prompt_list = {"b":prompt_b}
    data_list = ["ecare", "musique", "squad"]
    mode_setting = ["no_context", "neg_context", "pos_context"]
    #mode_setting = ["neg_context", "pos_context"]
    config_setting = ["pos", "neg", "mix"]
    #answer_dict = {0: ' A: ', 1: ' B: ', 2: ' C: ', 3: " D: ", 4: ' E: ', 5: ' F: '}
    answer_dict = {0: ' A: ', 1: ' B: ', 2: ' C: ', 3: " D: ", 4: ' E: ', 5: ' F: '}
    for name in data_list:
        print("============================================dataset:", name)
        file_list = {}
        if name == "squad" or name == "musique":
            file_right = json.load(open("data/" + name + "/" + name + "_right_with_neg_fixed.json","r"))
            file_wrong = json.load(open("data/" + name + "/" + name + "_wrong_with_neg_fixed.json","r"))
        else:
            file_right = json.load(open("data/" + name + "/" + name + "_right_with_neg.json","r"))
            file_wrong = json.load(open("data/" + name + "/" + name + "_wrong_with_neg.json","r"))
        file_list["right"] = file_right
        file_list["wrong"] = file_wrong
        for prompt_name in list(prompt_list.keys()):
            print("---------------prompt:", prompt_name)
            prompt_e = prompt_list[prompt_name]
            for dataset_name in list(file_list.keys()):
                print("----------------------------right:", dataset_name)
                file = file_list[dataset_name]
                print(len(file))
                for mode in mode_setting:
                    print("----------------------mode:", mode)
                #print("------------------ mode: ",  mode)
                    if dataset_name == "wrong" and mode == "no_context":
                        continue
                    if dataset_name == "wrong" and mode == "neg_context":
                        continue
                    if dataset_name == "right" and mode == "pos_context":
                        continue
                    for config in config_setting:
                        if name == "ecare" and prompt_name == "a":
                            continue
                        if name == "ecare" and prompt_name == "b" and  dataset_name == "right" and mode =="no_context" :
                            if config =="pos" or config =="neg":
                                print(config, config, config)
                                continue
                        result = []
                        print("-----------------------config:", config)
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
                            prompt_example = prompt_example.replace(token_string, example_test_list[i-1])
                            if token_string in prompt_example:
                                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!context!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        ################################################
                        batch_case = []
                        batch_size = 4
                        index = 0
                        while index + batch_size < len(file):
                            sample = []
                            for i in range (batch_size):
                                sample.append(file[index])
                                index += 1
                            batch_case.append(sample)
                        sample = []
                        while index < len(file):
                            sample.append(file[index])
                            index += 1
                        batch_case.append(sample)
                        ################################################
                        length = len(batch_case)
                        bar = tqdm.trange(length)
                        #print(length)
                        for idx, (batch) in zip(bar, batch_case):
                            prompts_temps = []
                            for data in batch:
                                ###############################################
                                question = data["question"]
                                answer = data["answer"]
                                context = ""
                                if dataset_name == "wrong":
                                    context = data[" golden_context"]
                                elif dataset_name == "right":
                                    context = data["negative_context"]
                                else:
                                    print("Error")
                                choices = data["choices"]
                                choice = ""
                                for index, choi in enumerate(choices):
                                    choice += answer_dict[index] + choi
                                #################################################
                
                                prompts_temp = ""
                                if mode == "no_context":
                                    prompts_temp = prompt + "\n" + prompt_example +  "\n" + "Question: " + question + choice + "\n"
                                else:
                                    prompts_temp = prompt + "\n" + prompt_example +  "\n" +  "Context: " + context + "\n" + "Question: " + question + choice + "\n"
                                    #print("===========", prompts_temp)
                                #prompts_temp += prompt_example
                                prompts_temps.append(prompts_temp)
                                #print(prompts_temp)
                                
                            #print(prompts_temps)
                            #return
                            responses = generator.generate(
                                prompts_temps, max_gen_len=258, temperature=temperature, top_p=top_p
                            )

                            for index, res in enumerate(responses):
                                response = res[len(prompts_temps[index]):]
                            
                                sample = copy.deepcopy(batch[index])
                                sample["response"] = response
                                sample["mode"] = mode
                                sample["config"] = config
                                
                                result.append(sample)
                            ##################################################################3
                        json_data = json.dumps(result, sort_keys= False, ensure_ascii=False, indent=4, separators=(',', ': '))
                        file_write = open("result/" + name + "/vicuna/" + name + "_" + prompt_name +  "_" + dataset_name + "_" + mode + "_" + config + "_example.json","w", encoding='utf-8')
                        file_write.write(json_data)
                        print("============out", "result/" + name + "/vicuna/" + name + "_" + prompt_name +  "_" + dataset_name + "_" + mode + "_" + config + "_example.json")
                        #break



if __name__ == "__main__":
    fire.Fire(main)
