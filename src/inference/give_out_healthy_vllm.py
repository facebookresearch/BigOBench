# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from fire import Fire
from openai import OpenAI
import concurrent.futures

def create_completion_request(host, model):
    try:
        client = OpenAI(base_url=f"http://{host}:8000/v1", api_key="EMPTY")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Hi !",
            },
        ]

        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=2048, temperature=0.6, top_p=0.95, 
        )

        print('response obtained', host)

        return 200 if len(response.choices[0].message.content) >= 0 else 500

    except Exception as e:
        print(f"Error: {e}")
        return 500

def main(host_list: str, model: str, timeout: int = 240):
    print('model:', model)

    host_list = host_list.split(',')

    print('looking at', len(host_list), 'different hosts')

    if len(host_list) == 0:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(host_list)) as executor:
        # Submit the requests to the executor
        futures = [executor.submit(create_completion_request, host, model) for host in host_list]

        # Collect the results from the futures -> we really need to improve that
        done, not_done = concurrent.futures.wait(futures, timeout=timeout)

        print('finished', len(done))
        print('did not finish', len(not_done))

        for future in not_done:
            future.cancel()

        results = []

        for future in futures:
            try:
                result = future.result()
            except concurrent.futures.CancelledError:
                result = 0
            except Exception as e:
                # Handle any other exceptions
                print(f"Exception: {e}")
                result = 0
            results.append(result)

    healthy_host_list = []
    for host, result in zip(host_list, results):
        print(host, ":", result)
        if result == 200:
            healthy_host_list.append(host)

    print('Healthy hosts are:')
    print(",".join(healthy_host_list))

if __name__ == "__main__":
    Fire(main)
