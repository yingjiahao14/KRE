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

    prompt_a = open("data/prompt/a/" + "6" + ".txt", "r").read()
    prompt_b = open("data/prompt/b/" + "1" + ".txt", "r").read()
    prompt_list = {"a": prompt_a, "b":prompt_b}
    #prompt_list = {"b":prompt_b}
    data_list = ["musique", "squad"]
    answer_dict = {0: ' A: ', 1: ' B: ', 2: ' C: ', 3: " D: ", 4: ' E: ', 5: ' F: '}
    for name in data_list:
        print("============================================dataset:", name)
        file_list = {}
        file_right = json.load(open("data/" + name + "/" + name + "_right_with_neg_fixed.json","r"))
        file_wrong = json.load(open("data/" + name + "/" + name + "_wrong_with_neg_fixed.json","r"))
        file_list["right"] = file_right
        file_list["wrong"] = file_wrong
        for dataset_name in list(file_list.keys()):
            print("----------------------------right:", dataset_name)
            file = file_list[dataset_name]
            #print(len(file))
            for prompt_name in list(prompt_list.keys()):
                print("---------------prompt:", prompt_name)
                prompt = prompt_list[prompt_name]
                #print(prompt)
                ################################################
                batch_case = []
                index = 0
                while index + 12 < len(file):
                    sample = []
                    for i in range (12):
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
                result = []
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
        
                        prompts_temp = prompt + "\n" + "Context: " + context + "\n" + "Question: " + question + choice + "\n"
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
                        
                        result.append(sample)
                    ##################################################################3
                json_data = json.dumps(result, sort_keys= False, ensure_ascii=False, indent=4, separators=(',', ': '))
                file_write = open("result/" + name + "/vicuna/" + name + "_" + dataset_name + "_" + prompt_name + "_fixed.json","w", encoding='utf-8')
                file_write.write(json_data)



if __name__ == "__main__":
    fire.Fire(main)
