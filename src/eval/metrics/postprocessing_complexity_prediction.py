# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "../../../")
sys.path.insert(0, src_dir)

from fire import Fire
import collections
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

from src.eval.metrics.utils import (
    read_jsonl_file,
    correct_complexity_formatting,
    save_dict_to_json,
    aggregate_at_k,
    avg_dict,
    aggregate_dict_list_into_list_dict,
    pass_at_k,
)


from src.complexity.curve_fitting.fitting_utils import (
    equality_complexities,
    get_number_variables,
    get_complexity_order,
)

import numpy as np

def average_metrics_with_question_name_complexity_support(
    question_name_complexity_to_agg_metrics_dict,
    question_name_complexity_set,
    metric_name_at_list,
):
    result_dict = {}

    question_name_set = set(x[0] for x in question_name_complexity_set)
    question_name_to_complexity_dict = collections.defaultdict(list)
    for question_name, complexity in question_name_complexity_set:
        question_name_to_complexity_dict[question_name].append(complexity)

    assert set(question_name_to_complexity_dict.keys()) == question_name_set

    for metric_name in metric_name_at_list:
        value_list = []

        for question_name in question_name_set:
            nested_value_list = list(
                agg_metrics_dict[metric_name]
                for complexity, agg_metrics_dict in list(question_name_complexity_to_agg_metrics_dict[question_name].items())
                if any(
                    equality_complexities(complexity, complexity_2) 
                    for complexity_2 in question_name_to_complexity_dict[question_name]
                )
            )
            assert len(nested_value_list) == len(question_name_to_complexity_dict[question_name])
            try:
                if 'list' not in metric_name:
                    assert type(nested_value_list[0]) != str and type(nested_value_list[0]) != list
                    float(nested_value_list[0])
                    value_list.append(
                        float(np.mean(nested_value_list)) if len(nested_value_list) > 0 else 0.0
                    )
                else:
                    value_list.append(
                        float(np.mean([float(np.mean(nested_value_list_temp)) for nested_value_list_temp in nested_value_list])) 
                        if len(nested_value_list) > 0 
                        else 0.0
                    )
            except:
                print(metric_name)
                print(nested_value_list)
                raise Exception()

        assert len(value_list) == len(question_name_set)
        result_dict[metric_name] = float(np.mean(value_list)) if len(value_list) > 0 else 0.0

    return result_dict

# first we do the metrics by just enforcing the common question_names !

# best is the time@ for the best complexity
# all_time is the time@ at the same time for all complexity 
# non_best 

def get_best_and_non_best_and_all_pass_at(
    question_name_complexity_to_agg_metrics_dict,
    question_name_several_complexity_set,
    metric_name,
    pass_at_list,
):
    assert metric_name in ['time', 'space', 'pass']
    
    result_dict = {}

    question_name_set = set(x[0] for x in question_name_several_complexity_set)

    question_name_to_complexity_dict = collections.defaultdict(list)
    for question_name, complexity in question_name_several_complexity_set:
        question_name_to_complexity_dict[question_name].append(complexity)

    for question_name in question_name_to_complexity_dict.keys():
        assert len(question_name_to_complexity_dict[question_name]) >= 2

    question_name_to_best_complexity_dict = collections.defaultdict(list)
    question_name_to_non_best_complexity_dict = collections.defaultdict(list)

    for question_name, complexity_list in question_name_to_complexity_dict.items():
        complexity_list_sorted = sorted(complexity_list, key = lambda x: (get_complexity_order(x), get_number_variables(x)))
        assert len(complexity_list_sorted) >= 2
        question_name_to_best_complexity_dict[question_name] = complexity_list_sorted[0]
        question_name_to_non_best_complexity_dict[question_name] = complexity_list_sorted[1:]

    assert set(question_name_to_complexity_dict.keys()) == question_name_set
    assert set(question_name_to_best_complexity_dict.keys()) == question_name_set
    assert set(question_name_to_non_best_complexity_dict.keys()) == question_name_set

    for k in pass_at_list:
        value_list = []

        for question_name in question_name_set:
            nested_value_list = [
                agg_metrics_dict[f'{metric_name}_at_{k}']
                for complexity, agg_metrics_dict in list(question_name_complexity_to_agg_metrics_dict[question_name].items())
                if equality_complexities(complexity, question_name_to_best_complexity_dict[question_name]) 
            ]

            assert len(nested_value_list) == 1
            assert type(nested_value_list[0]) != str and type(nested_value_list[0]) != list
            float(nested_value_list[0])
            value_list.append(
                nested_value_list[0]
            )

        assert len(value_list) == len(question_name_set)
        result_dict[f'best_{metric_name}_at_{k}'] = float(np.mean(value_list)) if len(value_list) > 0 else 0.0

    for k in pass_at_list:
        value_list = []

        for question_name in question_name_set:
            nested_value_list = [
                agg_metrics_dict[f'{metric_name}_at_{k}']
                for complexity, agg_metrics_dict in list(question_name_complexity_to_agg_metrics_dict[question_name].items())
                if any(
                    equality_complexities(complexity, complexity_2)
                    for complexity_2 in question_name_to_non_best_complexity_dict[question_name]
                )
            ]
            assert len(nested_value_list) >= 1
            assert type(nested_value_list[0]) != str and type(nested_value_list[0]) != list
            float(nested_value_list[0])
            value_list.append(
                float(np.mean(nested_value_list)) if len(nested_value_list) > 0 else 0.0
            )

        assert len(value_list) == len(question_name_set)
        result_dict[f'nonbest_{metric_name}_at_{k}'] = float(np.mean(value_list)) if len(value_list) > 0 else 0.0

    for k in pass_at_list:
        value_list = []

        for question_name in question_name_set:
            nested_value_list = [
                agg_metrics_dict[f'list_{metric_name}']
                for complexity, agg_metrics_dict in list(question_name_complexity_to_agg_metrics_dict[question_name].items())
                if any(
                    equality_complexities(complexity, complexity_2)
                    for complexity_2 in question_name_to_complexity_dict[question_name]
                )
            ]
            assert len(nested_value_list) >= 2
            for x in nested_value_list:
                for y in x:
                    assert y == 100 or y == 0

            flatten_nested_value_list = []

            for i in range(min(len(nested_value_list_temp) for nested_value_list_temp in nested_value_list)):
                flatten_nested_value_list.append(
                    100 if all([x[i] == 100 for x in nested_value_list]) else 0
                )

            value_list.append(
                pass_at_k(len(flatten_nested_value_list), sum(flatten_nested_value_list) / 100, k=k) * 100
            )

        assert len(value_list) == len(question_name_set)
        result_dict[f'all_{metric_name}_at_{k}'] = float(np.mean(value_list)) if len(value_list) > 0 else 0.0

    return result_dict

def main(
    results_file_path: str,
    dump_dir: str,
    test_set_file_path: str,
    time_or_space: str,
    generate_baseline: bool = False,
    at_k: int = 10, 
):
    n_samples = 2 * at_k if at_k > 1 else 1
    pass_ats = [1,]
    assert time_or_space in ["time", "space"]

    test_set = read_jsonl_file(test_set_file_path)
    question_name_complexity_set = set(
        (x["problem_name"], x[f"{time_or_space}_complexity_inferred"]) for x in test_set
    )

    question_name_complexity_set = set(
        (x, correct_complexity_formatting(y)) for x,y in question_name_complexity_set
    )

    all_full_question_list = read_jsonl_file(results_file_path)

    eval_question_name_complexity_set = set(
        (x["problem_name"], x[f"{time_or_space}_complexity_synthetic_ground_truth"]) for x in all_full_question_list
    )

    if generate_baseline:
        print('generating baseline')
        for i in range(len(all_full_question_list)):
            all_full_question_list[i]['infered_complexity'] = 'o(n)'

            all_full_question_list[i]["pass_at_1"] = 100.0 * int(equality_complexities(
                all_full_question_list[i]['infered_complexity'], 
                all_full_question_list[i][f"{time_or_space}_complexity_synthetic_ground_truth"]
            ))

    # Check whether the whole test set is covered

    if eval_question_name_complexity_set != question_name_complexity_set:
        print("\n!!Careful: the model has not generated results on the entire test set, so results will be partial\n")

    print(f"\nGenerating results on {len(set(x for x,_ in eval_question_name_complexity_set))} problems\n")

    parsed_complexity_list = [
        x["infered_complexity"] for x in all_full_question_list
    ]

    true_complexity_list = [
        x[f"{time_or_space}_complexity_synthetic_ground_truth"] for x in all_full_question_list
    ]

    # We can make sure though that we retrieve what was supposed to be the question_name, time and space that we were expecting

    task_id_to_question_name_dict = collections.defaultdict(list)
    task_id_to_time_dict = collections.defaultdict(list)
    task_id_to_space_dict = collections.defaultdict(list)
    task_id_to_dataclass_code_dict = collections.defaultdict(list)
    task_id_to_variable_name_to_input_dict_dict = collections.defaultdict(list)

    for x in all_full_question_list:
        task_id_to_question_name_dict[x['raw']['task_id']].append(x["problem_name"])
        task_id_to_time_dict[x['raw']['task_id']].append(x["time_complexity_synthetic_ground_truth"])
        task_id_to_space_dict[x['raw']['task_id']].append(x["space_complexity_synthetic_ground_truth"])
        task_id_to_dataclass_code_dict[x['raw']['task_id']].append(x["dataclass_code"])
        task_id_to_variable_name_to_input_dict_dict[x['raw']['task_id']].append(x["inputs_example"])

    assert len(task_id_to_question_name_dict) == len(task_id_to_time_dict)
    assert len(task_id_to_question_name_dict) == len(task_id_to_space_dict)
    assert len(task_id_to_question_name_dict) == len(task_id_to_dataclass_code_dict)
    assert len(task_id_to_question_name_dict) == len(task_id_to_variable_name_to_input_dict_dict)

    assert set(task_id_to_question_name_dict.keys()) == set(task_id_to_time_dict.keys())
    assert set(task_id_to_question_name_dict.keys()) == set(task_id_to_space_dict.keys())
    assert set(task_id_to_question_name_dict.keys()) == set(task_id_to_dataclass_code_dict.keys())
    assert set(task_id_to_question_name_dict.keys()) == set(task_id_to_variable_name_to_input_dict_dict.keys())

    for key_ in task_id_to_question_name_dict.keys():
        value_set = set(task_id_to_question_name_dict[key_])

        if len(value_set) == 1:
            continue
        elif len(value_set) == 2:
            raise Exception('weird')
            assert '' in value_set
        else:
            raise Exception('not covered')
        
    for key_ in task_id_to_time_dict.keys():
        value_set = set(task_id_to_time_dict[key_])

        if len(value_set) == 1:
            continue
        elif len(value_set) == 2:
            raise Exception('weird')
            assert '' in value_set
        else:
            raise Exception('not covered')
        
    for key_ in task_id_to_space_dict.keys():
        value_set = set(task_id_to_space_dict[key_])

        if len(value_set) == 1:
            continue
        elif len(value_set) == 2:
            raise Exception('weird')
            assert '' in value_set
        else:
            raise Exception('not covered')
        
    for key_ in task_id_to_dataclass_code_dict.keys():
        value_set = set(task_id_to_dataclass_code_dict[key_])

        if len(value_set) == 1:
            continue
        elif len(value_set) == 2:
            raise Exception('weird')
            assert '' in value_set
        else:
            raise Exception('not covered')
        
    for key_ in task_id_to_variable_name_to_input_dict_dict.keys():
        value_set = set(task_id_to_variable_name_to_input_dict_dict[key_])

        if len(value_set) == 1:
            continue
        elif len(value_set) == 2:
            raise Exception('weird')
            assert '' in value_set
        else:
            raise Exception('not covered')

    for i, x in enumerate(all_full_question_list):

        value_set = set(task_id_to_question_name_dict[x['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('weird')
            value_set.remove('')
        if x["problem_name"] == '':
            raise Exception('weird')
            all_full_question_list[i]["problem_name"] = next(iter(value_set))

        value_set = set(task_id_to_time_dict[x['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('weird')
            value_set.remove('')
        if x["time_complexity_synthetic_ground_truth"] == '':
            raise Exception('weird')
            all_full_question_list[i]["time_complexity_synthetic_ground_truth"] = next(iter(value_set))

        value_set = set(task_id_to_space_dict[x['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('weird')
            value_set.remove('')
        if x["space_complexity_synthetic_ground_truth"] == '':
            raise Exception('weird')
            all_full_question_list[i]["space_complexity_synthetic_ground_truth"] = next(iter(value_set))

        value_set = set(task_id_to_dataclass_code_dict[x['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('weird')
            value_set.remove('')
        if x["dataclass_code"] == '':
            raise Exception('weird')
            all_full_question_list[i]["dataclass_code"] = next(iter(value_set))

        value_set = set(task_id_to_variable_name_to_input_dict_dict[x['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('weird')
            value_set.remove('')
        if x["inputs_example"] == '':
            raise Exception('weird')
            all_full_question_list[i]["inputs_example"] = next(iter(value_set))

    for x in all_full_question_list:
        if x["problem_name"] == "":
            count_ += 1
            raise Exception()


    is_success_list = []

    for parsed_complexity, true_complexity in zip(
        parsed_complexity_list, true_complexity_list
    ):
        if parsed_complexity is not None:
            is_success_list.append(equality_complexities(parsed_complexity, true_complexity))

        else:
            is_success_list.append(parsed_complexity == true_complexity)

    is_success_list = np.array(is_success_list)

    is_success_list_1 = is_success_list

    sum(is_success_list)/len(is_success_list)


    metric_name_list = [
        "pass",
    ]

    avg_metrics = None
        
    metric_dict = aggregate_dict_list_into_list_dict(all_full_question_list, metric_name_list)

    for full_question, value_ in zip(all_full_question_list, metric_dict["pass_at_1"]):
        assert value_ == full_question['pass_at_1']

    agg, metrics_to_task_id_dict = aggregate_at_k(
        metrics = metric_dict,
        at_metrics = metric_name_list,
        n_samples = n_samples,
        pass_ats = pass_ats,
        avg_metrics = avg_metrics,
        metrics_for_list = ["time", "space", "pass"],
    )

    avg_dict(
        [f"{metric_name}_at_{pass_at}" for metric_name in metric_name_list for pass_at in pass_ats] + [f"any_{metric_name}" for metric_name in metric_name_list], 
        agg, 
        dist=False
    )

    for key_1 in metrics_to_task_id_dict.keys():
        for key_2 in metrics_to_task_id_dict.keys():
            assert metrics_to_task_id_dict[key_1] == metrics_to_task_id_dict[key_2]


    task_id_to_question_name_dict = dict()
    for x in all_full_question_list:
        task_id_to_question_name_dict[x['raw']['task_id']] = x["problem_name"]

    task_id_to_complexity_dict = dict()
    for x in all_full_question_list:
        if x['raw']['task_id'] not in task_id_to_complexity_dict.keys():
            task_id_to_complexity_dict[x['raw']['task_id']] = x[f"{time_or_space}_complexity_synthetic_ground_truth"]
        else:
            assert task_id_to_complexity_dict[x['raw']['task_id']] == x[f"{time_or_space}_complexity_synthetic_ground_truth"]


    # Not the right format for export and further processing

    task_id_set = set([
        task_id 
        for task_id, complexity in list(task_id_to_complexity_dict.items())
        if complexity not in [None, 'none', 'None', '']
    ])

    assert all(type(x) is str and len(x) > 2 for x in (task_id_to_question_name_dict[task_id] for task_id in task_id_set))

    question_name_complexity_to_agg_metrics_dict = collections.defaultdict(dict)

    for task_id in task_id_set:
        question_name = task_id_to_question_name_dict[task_id]
        complexity = task_id_to_complexity_dict[task_id]

        index_mem = None

        agg_metrics = {}

        for metric_name in metrics_to_task_id_dict.keys():
            try:
                index = metrics_to_task_id_dict[metric_name].index(task_id)

                if index_mem is None:
                    index_mem = index
                else:
                    assert index_mem == index

                agg_metrics[metric_name] = agg[metric_name][index]
            except:
                index = (metrics_to_task_id_dict)[metric_name].index(task_id)

                if index_mem is None:
                    index_mem = index
                else:
                    assert index_mem == index

                agg_metrics[metric_name] = (agg)[metric_name][index]

        question_name_complexity_to_agg_metrics_dict[question_name][complexity] = agg_metrics


    for question_name in question_name_complexity_to_agg_metrics_dict.keys():
        complexity_list = list(question_name_complexity_to_agg_metrics_dict[question_name].keys())
        complexity_list = sorted(complexity_list, key = get_number_variables)

        complexity_list_retained = []

        for complexity_1 in complexity_list:
            found = False

            for complexity_2 in complexity_list_retained:
                if equality_complexities(complexity_1, complexity_2):
                    found = True

            if not found:
                complexity_list_retained.append(complexity_1)

        for complexity in complexity_list:
            if complexity not in complexity_list_retained:
                raise Exception('weird ?')


    metric_name_at_list = list(metrics_to_task_id_dict.keys())

    temp_list = []

    for x in question_name_complexity_to_agg_metrics_dict.keys():
        for y in question_name_complexity_to_agg_metrics_dict[x].keys():
            temp_list.append((x,y))

    assert len(temp_list) == len(set(temp_list))

    assert set(temp_list) == eval_question_name_complexity_set


    question_name_several_set = (
        set(x for x,y in dict(collections.Counter(x[0] for x in list(eval_question_name_complexity_set))).items() if y > 1)
    )

    question_name_several_complexity_set = set(
        (x,y) for x,y in eval_question_name_complexity_set
        if x in question_name_several_set
    )

    assert question_name_several_complexity_set == eval_question_name_complexity_set

    result_dict_temp = (average_metrics_with_question_name_complexity_support(
        question_name_complexity_to_agg_metrics_dict,
        question_name_several_complexity_set,
        metric_name_at_list,
    ) | get_best_and_non_best_and_all_pass_at(
        question_name_complexity_to_agg_metrics_dict,
        question_name_several_complexity_set,
        'pass',
        [1],
    ))

    print("\n {} & {} & {} \n".format(
        round(result_dict_temp["pass_at_1"], 1),
        # round(result_dict_temp["pass_at_10"], 1),
        round(result_dict_temp["best_pass_at_1"], 1),
        # round(result_dict_temp["nonbest_pass_at_1"], 1),
        round(result_dict_temp["all_pass_at_1"], 1),
    ))

    save_dict_to_json(
        {
            f"task/complexity_prediction/{time_or_space}/pass_at_1": result_dict_temp["pass_at_1"],
            f"task/complexity_prediction/{time_or_space}/best_at_1": result_dict_temp["best_pass_at_1"],
            f"task/complexity_prediction/{time_or_space}/all_at_1": result_dict_temp["all_pass_at_1"],
        },
        os.path.join(dump_dir, "postprocessed_results.json")
    )

if __name__ == "__main__":
    Fire(main)