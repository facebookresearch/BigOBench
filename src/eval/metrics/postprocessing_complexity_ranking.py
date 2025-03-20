# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from fire import Fire
import shutil
import os
import sys
import json
from tqdm import tqdm
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "../../../")
sys.path.insert(0, src_dir)
import collections

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

from src.complexity.utils import correct_complexity_formatting

from src.complexity.curve_fitting.fitting_utils import get_number_variables

# first we do the metrics by just enforcing the common question_names !

def average_metrics_with_question_name_support(
    question_name_complexity_to_agg_metrics_dict,
    question_name_set,
    metric_name_at_list,
):
    result_dict = {}

    for metric_name in metric_name_at_list:
        value_list = []

        for question_name in question_name_set:
            value_list.extend(
                list(
                    agg_metrics_dict[metric_name]
                    for agg_metrics_dict in list(question_name_complexity_to_agg_metrics_dict[question_name].values())
                )
            )

        assert len(value_list) > len(question_name_set)
        result_dict[metric_name] = float(np.mean(value_list)) if len(value_list) > 0 else 0.0

    return result_dict


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
            # value_list.append(
            #     float(np.mean(nested_value_list)) if len(nested_value_list) > 0 else 0.0
            # )

        assert len(value_list) == len(question_name_set)
        result_dict[metric_name] = float(np.mean(value_list)) if len(value_list) > 0 else 0.0

    return result_dict


def get_best_and_non_best_and_all_pass_at(
    question_name_complexity_to_agg_metrics_dict,
    question_name_several_complexity_set,
    metric_name,
    pass_at_list,
):
    assert metric_name in ['time', 'space']
    
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
            # print(len(nested_value_list))
            # print(question_name_to_best_complexity_dict[question_name])
            # print(list(question_name_complexity_to_agg_metrics_dict[question_name].keys()))
            assert len(nested_value_list) == 1
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


def percentile_metrics_with_question_name_complexity_support(
    question_name_complexity_to_agg_metrics_dict,
    question_name_complexity_set,
    metric_type,
    question_name_to_complexity_to_ordered_coeffs_dict,
):
    result_list = []
    
    question_name_set = set(x[0] for x in question_name_complexity_set)
    question_name_to_complexity_dict = collections.defaultdict(list)
    for question_name, complexity in question_name_complexity_set:
        question_name_to_complexity_dict[question_name].append(complexity)

    assert set(question_name_to_complexity_dict.keys()) == question_name_set

    def get_percentile(
        question_name, 
        complexity, 
        coeff,
    ):
        i = 0
        while i < len(question_name_to_complexity_to_ordered_coeffs_dict[question_name][complexity]):
            human_coeff = question_name_to_complexity_to_ordered_coeffs_dict[question_name][complexity][i]
            if coeff > human_coeff:
                break
            i += 1

        # for i, human_coeff in enumerate(question_name_to_complexity_to_ordered_coeffs_dict[question_name][complexity]):
        #     if coeff > human_coeff:
        #         break
        # else:
        #     i += 1
        return i / len(question_name_to_complexity_to_ordered_coeffs_dict[question_name][complexity])

    for question_name in question_name_set:
        nested_value_list = list(
            (
                get_percentile(
                    question_name, 
                    complexity, 
                    agg_metrics_dict[f'{metric_type}_coeff'],
                )
                if agg_metrics_dict[f'{metric_type}_coeff'] is not None
                else 0
            )
            for complexity, agg_metrics_dict in list(question_name_complexity_to_agg_metrics_dict[question_name].items())
            if any(
                equality_complexities(complexity, complexity_2) 
                for complexity_2 in question_name_to_complexity_dict[question_name]
            )
        )
        assert len(nested_value_list) == len(question_name_to_complexity_dict[question_name])
        result_list.append(
            float(np.mean(nested_value_list)) if len(nested_value_list) > 0 else 0.0
        )

    assert len(result_list) == len(question_name_set)
    result = float(np.mean(result_list)) if len(result_list) > 0 else 0.0

    return result

def main(
    results_folder_or_file_path: str,
    dump_dir: str,
    test_set_file_path: str,
    time_or_space: str,
    generate_baseline: bool = False,
    unzip_files: bool = True,    
    at_k: int = 10, 
):
    n_samples = 2 * at_k if at_k > 1 else 1
    pass_ats = [x for x in [1, 2, 5, 10] if 2*x <= n_samples]
    assert time_or_space in ["time", "space"]

    test_set = read_jsonl_file(test_set_file_path)
    question_name_complexity_set = set(
        (x["problem_name"], x[f"{time_or_space}_complexity_inferred"]) for x in test_set
    )

    question_name_complexity_set = set(
        (x, correct_complexity_formatting(y)) for x,y in question_name_complexity_set
    )

    subfolder_sorting_function = lambda x: str(x)

    if unzip_files:
        print('unzipping')
        import glob, os

        if os.path.isfile(results_folder_or_file_path):
            shutil.unpack_archive(results_folder_or_file_path, os.path.dirname(results_folder_or_file_path), 'zip')

        else:
            for file_name in glob.glob(os.path.join(results_folder_or_file_path, '**/*.zip'), recursive=True):
                shutil.unpack_archive(file_name, results_folder_or_file_path, 'zip')

    import glob, os

    subfolder_name_set = set()

    if os.path.isfile(results_folder_or_file_path):
        print(results_folder_or_file_path)
        print(os.path.splitext(results_folder_or_file_path))
        subfolder_name_set = set(
            [os.path.splitext(results_folder_or_file_path)[0]]
        )
    else:
        for file_name in glob.glob(os.path.join(
            results_folder_or_file_path, 
            '**/*.json'
        ), recursive=True):
            subfolder_name_set.add('/'.join(file_name.split('/')[:-1]))

    subfolder_name_set = sorted(list(subfolder_name_set), key = subfolder_sorting_function)

    print(subfolder_name_set)

    complexity_labels_light = []

    for subfolder_name in tqdm(subfolder_name_set):
        with open(os.path.join(subfolder_name, 'complexity_labels_light.json'), 'r') as f:
            for x in json.loads(f.read()):
                complexity_labels_light.append(x)

    question_name_list = [
        x["problem_name"] for x in complexity_labels_light
    ]

    eval_question_name_complexity_set = set(
        (
            x["problem_name"], 
            x['additional_data_from_input'][f"{time_or_space}_complexity_synthetic_ground_truth"]
        ) for x in complexity_labels_light
    )

    if eval_question_name_complexity_set != question_name_complexity_set:
        print("\n!!Careful: the model has not generated results on the entire test set, so results will be partial\n")

    print(f"\nGenerating results on {len(set(x for x,_ in eval_question_name_complexity_set))} problems\n")

    counter_total = 0
    counter_local = 0

    for complexity_label in complexity_labels_light:
        counter_total += 1

        if complexity_label["additional_data_from_input"]['dialog'][1]['body'] == '':
            counter_local += 1

    print('empty:', counter_local, 'out of', counter_total)
    print(
        "number of tasks:", 
        len(set((x["additional_data_from_input"]['raw']['task_id'] for x in complexity_labels_light)))
    )

    task_id_to_question_name_dict = collections.defaultdict(list)
    task_id_to_complexity_dict = collections.defaultdict(list)
    task_id_to_dataclass_code_dict = collections.defaultdict(list)
    task_id_to_variable_name_to_input_dict_dict = collections.defaultdict(list)

    for x in complexity_labels_light:
        task_id_to_question_name_dict[
            x["additional_data_from_input"]['raw']['task_id']
        ].append(x["problem_name"])

        task_id_to_complexity_dict[
            x["additional_data_from_input"]['raw']['task_id']
        ].append(x['additional_data_from_input'][f"{time_or_space}_complexity_synthetic_ground_truth"])

        task_id_to_dataclass_code_dict[
            x["additional_data_from_input"]['raw']['task_id']
        ].append(x['additional_data_from_input']["dataclass_code"])

        task_id_to_variable_name_to_input_dict_dict[
            x["additional_data_from_input"]['raw']['task_id']
        ].append(x['additional_data_from_input']["inputs_example"])

    assert len(task_id_to_question_name_dict) == len(task_id_to_complexity_dict)
    assert len(task_id_to_question_name_dict) == len(task_id_to_dataclass_code_dict)
    assert len(task_id_to_question_name_dict) == len(task_id_to_variable_name_to_input_dict_dict)

    assert set(task_id_to_question_name_dict.keys()) == set(task_id_to_complexity_dict.keys())
    assert set(task_id_to_question_name_dict.keys()) == set(task_id_to_dataclass_code_dict.keys())
    assert set(task_id_to_question_name_dict.keys()) == set(task_id_to_variable_name_to_input_dict_dict.keys())

    for key_ in task_id_to_question_name_dict.keys():
        value_set = set(task_id_to_question_name_dict[key_])

        if len(value_set) == 1:
            continue

        else:
            raise Exception('not covered')
        
    for key_ in task_id_to_complexity_dict.keys():
        value_set = set(task_id_to_complexity_dict[key_])

        if len(value_set) == 1:
            continue

        else:
            raise Exception('not covered')
        
    for key_ in task_id_to_dataclass_code_dict.keys():
        value_set = set(task_id_to_dataclass_code_dict[key_])

        if len(value_set) == 1:
            continue

        else:
            raise Exception('not covered')
        
    for key_ in task_id_to_variable_name_to_input_dict_dict.keys():
        value_set = set(task_id_to_variable_name_to_input_dict_dict[key_])

        if len(value_set) == 1:
            continue

        else:
            raise Exception('not covered')

    for i, x in enumerate(complexity_labels_light):

        value_set = set(task_id_to_question_name_dict[x["additional_data_from_input"]['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('are we sure this case is supposed to happen ?')

        if x["problem_name"] == '':
            raise Exception('are we sure this case is supposed to happen ?')

        value_set = set(task_id_to_complexity_dict[x["additional_data_from_input"]['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('are we sure this case is supposed to happen ?')

        if x['additional_data_from_input'][f"{time_or_space}_complexity_synthetic_ground_truth"] == '':
            raise Exception('are we sure this case is supposed to happen ?')

        value_set = set(task_id_to_dataclass_code_dict[x["additional_data_from_input"]['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('are we sure this case is supposed to happen ?')

        if x['additional_data_from_input']["dataclass_code"] == '':
            raise Exception('are we sure this case is supposed to happen ?')

        value_set = set(task_id_to_variable_name_to_input_dict_dict[x["additional_data_from_input"]['raw']['task_id']])
        if len(value_set) == 2:
            raise Exception('are we sure this case is supposed to happen ?')

        if x['additional_data_from_input']["inputs_example"] == '':
            raise Exception('are we sure this case is supposed to happen ?')

    for x in complexity_labels_light:
        if x["problem_name"] == "":
            count_ += 1
            raise Exception()
        
    for y in [
        x["additional_data_from_input"][f"{time_or_space}_complexity_synthetic_ground_truth"] 
        for x in complexity_labels_light
    ]:
        assert y is not None
        assert y.lower != 'none' and y != ""
    
    is_success_list = []

    for complexity_label in complexity_labels_light:
        parsed_complexity = complexity_label[f"{time_or_space}_complexity_inferred"]
        true_complexity = complexity_label[
            "additional_data_from_input"
        ][f"{time_or_space}_complexity_synthetic_ground_truth"] 

        if (
            parsed_complexity is not None 
            and true_complexity is not None 
            and true_complexity.lower != 'none' 
            and true_complexity != ""
        ):
            is_success_list.append(equality_complexities(parsed_complexity, true_complexity))

        else:
            is_success_list.append(False)

    is_success_list = np.array(is_success_list)

    print(sum(is_success_list)/len(is_success_list))

    # the reasoning capabilities poorly scale ! (or that means there is a problem and non solvable problems)
    for i in range(len(complexity_labels_light)):
        complexity_labels_light[i][f'{time_or_space}_at_1'] = 100 if (
            is_success_list[i] and complexity_labels_light[i]["additional_data_from_input"]['pass_at_1'] == 100
        ) else 0

        complexity_labels_light[i]['pass_at_1'] = complexity_labels_light[i][
            "additional_data_from_input"
        ]['pass_at_1']

        complexity_labels_light[i]['public_pass_at_1'] = complexity_labels_light[i][
            "additional_data_from_input"
        ]['public_pass_at_1']

        complexity_labels_light[i]['compiles_at_1'] = complexity_labels_light[i][
            "additional_data_from_input"
        ]['compiles_at_1']

        complexity_labels_light[i]['raw'] = complexity_labels_light[i][
            "additional_data_from_input"
        ]['raw']

    metric_name_list = [
        "pass",
        "public_pass",
        "compiles",
        time_or_space,
    ]

    avg_metrics = None
        
    metric_dict = aggregate_dict_list_into_list_dict(complexity_labels_light, metric_name_list)

    for complexity_label, value_ in zip(complexity_labels_light, metric_dict["pass_at_1"]):
        assert value_ == complexity_label['pass_at_1']

    # we need to train/test split etc + connect the metadata :)
    agg, metrics_to_task_id_dict = aggregate_at_k(
        metrics = metric_dict,
        at_metrics = metric_name_list,
        n_samples = n_samples,
        pass_ats = pass_ats,
        avg_metrics = avg_metrics,
    )

    avg_dict(
        [
            f"{metric_name}_at_{pass_at}" 
            for metric_name in metric_name_list 
            for pass_at in pass_ats
        ] + [
            f"any_{metric_name}" 
            for metric_name in metric_name_list
        ], 
        agg, 
        dist=False
    )

    for key_1 in metrics_to_task_id_dict.keys():
        for key_2 in metrics_to_task_id_dict.keys():
            # print(key_1, key_2)
            assert metrics_to_task_id_dict[key_1] == metrics_to_task_id_dict[key_2]

    if time_or_space == "time":
        for x in complexity_labels_light:
            assert "time_at_1" in x
            assert "space_at_1" not in x

    elif time_or_space == "space":
        for x in complexity_labels_light:
            assert "space_at_1" in x
            assert "time_at_1" not in x

    else:
        raise Exception('not handled')
    
    task_id_to_question_name_dict = dict()
    for x in complexity_labels_light:
        task_id_to_question_name_dict[x['raw']['task_id']] = x["problem_name"]

    task_id_to_complexity_dict = dict()
    for x in complexity_labels_light:
        if x['raw']['task_id'] not in task_id_to_complexity_dict.keys():
            task_id_to_complexity_dict[x['raw']['task_id']] = x[
                "additional_data_from_input"
            ][f"{time_or_space}_complexity_synthetic_ground_truth"] 
        else:
            assert task_id_to_complexity_dict[x['raw']['task_id']] == x[
                "additional_data_from_input"
            ][f"{time_or_space}_complexity_synthetic_ground_truth"] 

    assert all(x[f'{time_or_space}_at_1'] == 100.0 or x[f'{time_or_space}_at_1'] == 0 for x in complexity_labels_light)
    assert all(x[f'{time_or_space}_at_1'] <= x[f'{"pass"}_at_1'] for x in complexity_labels_light)
    assert all(x[f'{"pass"}_at_1'] <= x[f'{"public_pass"}_at_1'] for x in complexity_labels_light)
    assert all(x[f'{"public_pass"}_at_1'] <= x[f'{"compiles"}_at_1'] for x in complexity_labels_light)

    if time_or_space == "time":
        task_id_to_time_coeff_dict = collections.defaultdict(list)
        for x in complexity_labels_light:
            task_id_to_time_coeff_dict[x['raw']['task_id']].append(
                x["time_curve_coefficient"] if x['time_at_1'] > 50 else None
            )

    elif time_or_space == "space":
        task_id_to_space_coeff_dict = collections.defaultdict(list)
        for x in complexity_labels_light:
            task_id_to_space_coeff_dict[x['raw']['task_id']].append(
                x["space_curve_coefficient"] if x['space_at_1'] > 50 else None
            )
            
    else:
        raise Exception('not handled')

    assert '' not in task_id_to_question_name_dict.values()

    # Not the right format for export and further processing

    for task_id, label_ in list(task_id_to_complexity_dict.items()):
        if label_ in [None, 'none', 'None', '']:
            raise Exception('')

    task_id_set = set([
        task_id 
        for task_id, label_ in list(task_id_to_complexity_dict.items())
        if label_ not in [None, 'none', 'None', '']
    ])

    assert task_id_set == set(task_id_to_complexity_dict.keys())

    assert all(type(x) is str and len(x) > 2 for x in (task_id_to_question_name_dict[task_id] for task_id in task_id_set))

    question_name_label_to_agg_metrics_dict = collections.defaultdict(dict)

    for task_id in task_id_set:
        question_name = task_id_to_question_name_dict[task_id]
        label = (task_id_to_complexity_dict)[task_id]

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
                index = metrics_to_task_id_dict[metric_name].index(task_id)

                if index_mem is None:
                    index_mem = index
                else:
                    assert index_mem == index

                agg_metrics[metric_name] = agg[metric_name][index]

        if time_or_space == "time":
            agg_metrics["time_coeff"] = (
                min(x for x in task_id_to_time_coeff_dict[task_id] if x is not None) 
                if any(x is not None for x in task_id_to_time_coeff_dict[task_id])
                else None
            )
        elif time_or_space == "space":
            agg_metrics["space_coeff"] = (
                min(x for x in task_id_to_space_coeff_dict[task_id] if x is not None) 
                if any(x is not None for x in task_id_to_space_coeff_dict[task_id])
                else None
            )
        else:
            raise Exception('not handled')

        question_name_label_to_agg_metrics_dict[question_name][label] = agg_metrics
    # on enleve les dupplicate de complexities qui sont égales !

    for question_name in question_name_label_to_agg_metrics_dict.keys():
        complexity_list = list(question_name_label_to_agg_metrics_dict[question_name].keys())
        complexity_list = sorted(complexity_list, key = get_number_variables)

        complexity_list_retained = []

        for complexity_1 in complexity_list:
            found = False

            for complexity_2 in complexity_list_retained:
                if equality_complexities(complexity_1, complexity_2):
                    found = True
                    print('mmh some complexities are equivalent')

            if not found:
                complexity_list_retained.append(complexity_1)

        for complexity in complexity_list:
            if complexity not in complexity_list_retained:
                raise Exception('weird')

    metric_name_at_list = list(metrics_to_task_id_dict.keys())

    if unzip_files:
        print('deleting unzipped folder')

        for subfolder_name in subfolder_name_set:
            shutil.rmtree(subfolder_name)

    # question_name_label_to_agg_metrics_dict

    metric_type = time_or_space

    temp_list = []

    for x in question_name_label_to_agg_metrics_dict.keys():
        for y in question_name_label_to_agg_metrics_dict[x].keys():
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
        question_name_label_to_agg_metrics_dict,
        question_name_several_complexity_set,
        metric_name_at_list,
    ) | get_best_and_non_best_and_all_pass_at(
        question_name_label_to_agg_metrics_dict,
        question_name_several_complexity_set,
        time_or_space,
        [x for x in [1, 2, 5, 10] if 2*x <= n_samples],
    ))

    print(" {} & {} ".format(
        round(result_dict_temp["pass_at_1"], 1),
        round(result_dict_temp["pass_at_10"], 1) if 10 in pass_ats else None,
    ))

    print(" {} & {} & {} & {} ".format(
        round(result_dict_temp[f"{time_or_space}_at_1"], 1),
        round(result_dict_temp[f"{time_or_space}_at_10"], 1) if 10 in pass_ats else None,
        round(result_dict_temp[f"best_{time_or_space}_at_1"], 1),
        round(result_dict_temp[f"all_{time_or_space}_at_1"], 1),
    ))

    save_dict_to_json(
        {
            f"task/complexity_ranking/{time_or_space}/pass_at_1": result_dict_temp[f"pass_at_1"],
            f"task/complexity_ranking/{time_or_space}/pass_at_10": result_dict_temp[f"pass_at_10"] if 10 in pass_ats else None,
        },
        os.path.join(dump_dir, "postprocessed_results.json")
    )

    save_dict_to_json(
        {
            f"task/complexity_ranking/{time_or_space}/{time_or_space}_at_1": result_dict_temp[f"{time_or_space}_at_1"],
            f"task/complexity_ranking/{time_or_space}/{time_or_space}_at_10": result_dict_temp[f"{time_or_space}_at_10"] if 10 in pass_ats else None,
            f"task/complexity_ranking/{time_or_space}/best_{time_or_space}_at_1": result_dict_temp[f"best_{time_or_space}_at_1"],
            f"task/complexity_ranking/{time_or_space}/all_{time_or_space}_at_1": result_dict_temp[f"all_{time_or_space}_at_1"],
        },
        os.path.join(dump_dir, "postprocessed_results.json")
    )

    # And finally we just need to give the ranking metric !
    
    question_name_to_complexity_to_ordered_coeffs_dict = {}

    for x in test_set:
        if x["problem_name"] not in question_name_to_complexity_to_ordered_coeffs_dict:
            question_name_to_complexity_to_ordered_coeffs_dict[x["problem_name"]] = {}

        if x[f"{time_or_space}_complexity_inferred"] in question_name_to_complexity_to_ordered_coeffs_dict[x["problem_name"]]:
            raise Exception()
        
        question_name_to_complexity_to_ordered_coeffs_dict[x["problem_name"]][
            x[f"{time_or_space}_complexity_inferred"]
        ] = x[f"problem_{time_or_space}_curve_coefficient_list"]

        assert all([
                type(y) is float for y in question_name_to_complexity_to_ordered_coeffs_dict[x["problem_name"]][
                x[f"{time_or_space}_complexity_inferred"]
            ]
        ])

        def is_sorted_reverse(lst):
            return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))

        assert is_sorted_reverse([
                type(y) is float for y in question_name_to_complexity_to_ordered_coeffs_dict[x["problem_name"]][
                x[f"{time_or_space}_complexity_inferred"]
            ]
        ])

    for question_name, complexity in eval_question_name_complexity_set:
        assert len(question_name_to_complexity_to_ordered_coeffs_dict[question_name][complexity]) > 0

    print(f"""{round(100 * percentile_metrics_with_question_name_complexity_support(
        question_name_label_to_agg_metrics_dict,
        eval_question_name_complexity_set,
        metric_type,
        question_name_to_complexity_to_ordered_coeffs_dict
    ), 1)}""")

    save_dict_to_json(
        {
            f"task/complexity_ranking/{time_or_space}/coefficient_ranking_full": (
                100 * percentile_metrics_with_question_name_complexity_support(
                    question_name_label_to_agg_metrics_dict,
                    eval_question_name_complexity_set,
                    metric_type,
                    question_name_to_complexity_to_ordered_coeffs_dict
                )
            ),
        },
        os.path.join(dump_dir, "postprocessed_results.json")
    )

if __name__ == "__main__":
    Fire(main)