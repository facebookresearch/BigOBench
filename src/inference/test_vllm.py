# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from fire import Fire
from openai import OpenAI
import multiprocessing
import concurrent.futures

# legacy script, to play a bit with a server that was launched according to the readme.
# Just give the host as a string (node ID of one of the nodes the VLLM server is running on)
# along with the name of the model for instance "meta-llama/Meta-Llama-3.1-405B-Instruct"
def create_completion(host, model, messages):
    client = OpenAI(base_url=f"http://{host}:8000/v1", api_key="EMPTY")
    completion = client.chat.completions.create(
        model=model, messages=messages, max_tokens=2048
    )
    return completion
    
def main(host: str, model: str, number_queries: int = 1):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Can you write a poem in 10 words ?",
        },
    ]
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(create_completion, host, model, messages) for _ in range(number_queries)]
        for i, future in enumerate(futures):
            results.append(future.result())

    # with multiprocessing.Pool() as pool:
    #     results = pool.starmap(create_completion, [(host, model, messages)] * number_queries)
    
    for i, completion in enumerate(results):
        print(f"Prompt {i+1}:\n{messages[-1]['content']}")
        print("Response:\n")
        print(completion.choices[0].message.content)
        print("\n")


if __name__ == "__main__":
    Fire(main)
