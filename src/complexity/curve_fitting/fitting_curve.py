# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .fitting_tree import ComplexityCandidate, NodeComplexity, GroupComplexity
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import statistics
import sys


def is_there_peak(x_list, y_list):
    """
    Checks if there is a peak in the given data.
    A peak is defined as a point where the y-value is greater than the next y-value,
    but only considers points where the corresponding x-value is 256 or greater.
    Args:
        x_list (list): A list of x-values.
        y_list (list): A list of y-values corresponding to the x-values.
    Returns:
        bool: True if a peak is found, False otherwise.
    Raises:
        ValueError: If the input lists are not of the same length.
    """

    found_peak = False

    for i in range(len(y_list) - 1):
        if x_list[i] < 256:
            continue

        if y_list[i] > y_list[i+1]:
            found_peak = True

    return found_peak


def min_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    This function aggregates the minimum values from a dictionary of value lists 
    and returns two lists: one for the multiplier values (x_list) and one for the 
    corresponding minimum aggregated values (y_list).
    Args:
        multiplier_value_to_value_details_dict (dict): A dictionary where keys are 
            multiplier values and values are dictionaries containing a 'value_list' key.
        enlarge_values (bool): A boolean indicating whether to multiply the minimum 
            aggregated values by 10000.
    Returns:
        tuple: Two lists, x_list and y_list. x_list contains the multiplier values as integers, 
            and y_list contains the corresponding minimum aggregated values.
    Raises:
        Exception: If any value in the 'value_list' is None.
    """

    x_list, y_list = [], []

    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        if any([(value is None) for value in value_details['value_list']]):
            raise Exception('not supposed to be none')

        if len(value_details['value_list']) == 0:
            continue

        y_list.append(min(value_details['value_list'])
                      * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))

    return x_list, y_list


def max_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    This function aggregates the maximum values from a dictionary of value lists 
    and returns two lists: one for the multiplier values (x_list) and one for the 
    corresponding maximum aggregated values (y_list).
    Args:
        multiplier_value_to_value_details_dict (dict): A dictionary where keys are 
            multiplier values and values are dictionaries containing a 'value_list' key.
        enlarge_values (bool): A boolean indicating whether to multiply the maximum 
            aggregated values by 10000.
    Returns:
        tuple: Two lists, x_list and y_list. x_list contains the multiplier values as integers, 
            and y_list contains the corresponding maximum aggregated values.
    Raises:
        Exception: If any value in the 'value_list' is None.
    """

    x_list, y_list = [], []

    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        if any([(value is None) for value in value_details['value_list']]):
            raise Exception('not supposed to be none')

        if len(value_details['value_list']) == 0:
            continue

        y_list.append(max(value_details['value_list'])
                      * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))

    return x_list, y_list


def mean_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    This function aggregates the mean values from a dictionary of value lists 
    and returns two lists: one for the multiplier values (x_list) and one for the 
    corresponding mean aggregated values (y_list).
    Args:
        multiplier_value_to_value_details_dict (dict): A dictionary where keys are 
            multiplier values and values are dictionaries containing a 'value_list' key.
        enlarge_values (bool): A boolean indicating whether to multiply the mean 
            aggregated values by 10000.
    Returns:
        tuple: Two lists, x_list and y_list. x_list contains the multiplier values as integers, 
            and y_list contains the corresponding mean aggregated values.
    Raises:
        Exception: If any value in the 'value_list' is None.
    """

    x_list, y_list = [], []

    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        if any([(value is None) for value in value_details['value_list']]):
            raise Exception('not supposed to be none')

        if len(value_details['value_list']) == 0:
            continue

        y_list.append(
            np.mean(value_details['value_list']) * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))

    return x_list, y_list


def median_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    This function aggregates the median values from a dictionary of value lists 
    and returns two lists: one for the multiplier values (x_list) and one for the 
    corresponding median aggregated values (y_list).
    Args:
        multiplier_value_to_value_details_dict (dict): A dictionary where keys are 
            multiplier values and values are dictionaries containing a 'value_list' key.
        enlarge_values (bool): A boolean indicating whether to multiply the median 
            aggregated values by 10000.
    Returns:
        tuple: Two lists, x_list and y_list. x_list contains the multiplier values as integers, 
            and y_list contains the corresponding median aggregated values.
    Raises:
        Exception: If any value in the 'value_list' is None.
    """

    x_list, y_list = [], []

    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        if any([(value is None) for value in value_details['value_list']]):
            raise Exception('not supposed to be none')

        if len(value_details['value_list']) == 0:
            continue

        y_list.append(statistics.median(
            value_details['value_list']) * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))

    return x_list, y_list


def first_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    This function aggregates the first values from a dictionary of value lists 
    and returns two lists: one for the multiplier values (x_list) and one for the 
    corresponding first aggregated values (y_list).
    Args:
        multiplier_value_to_value_details_dict (dict): A dictionary where keys are 
            multiplier values and values are dictionaries containing a 'value_list' key.
        enlarge_values (bool): A boolean indicating whether to multiply the first 
            aggregated values by 10000.
    Returns:
        tuple: Two lists, x_list and y_list. x_list contains the multiplier values as integers, 
            and y_list contains the corresponding first aggregated values.
    Raises:
        Exception: If any value in the 'value_list' is None.
    """

    x_list, y_list = [], []

    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        if any([(value is None) for value in value_details['value_list']]):
            raise Exception('not supposed to be none')

        if len(value_details['value_list']) == 0:
            continue

        y_list.append((value_details['value_list'][0])
                      * (10000 if enlarge_values else 1))
        x_list.append(int(multiplier_value))

    return x_list, y_list


def most_stable_run_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    This function aggregates the values from a dictionary of value lists 
    and returns two lists: one for the multiplier values (x_list) and one for the 
    corresponding aggregated values (y_list). The function attempts to find the 
    most stable run by checking each individual run for a peak. If no peak is found, 
    it defaults to the median aggregate.
    Args:
        multiplier_value_to_value_details_dict (dict): A dictionary where keys are 
            multiplier values and values are dictionaries containing a 'value_list' key.
        enlarge_values (bool): A boolean indicating whether to multiply the aggregated 
            values by 10000.
    Returns:
        tuple: Two lists, x_list and y_list. x_list contains the multiplier values as integers, 
            and y_list contains the corresponding aggregated values.
    Raises:
        Exception: If any value in the 'value_list' is None.
    """

    number_runs = len(
        list(multiplier_value_to_value_details_dict.values())[0]['value_list'])
    x_list_list = [[] for _ in range(number_runs)]
    y_list_list = [[] for _ in range(number_runs)]

    for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
        if any([(value is None) for value in value_details['value_list']]):
            raise Exception('not supposed to be none')

        if len(value_details['value_list']) == 0:
            continue

        for i, value in enumerate(value_details['value_list']):
            if i >= number_runs:
                continue

            x_list_list[i].append(int(multiplier_value))
            y_list_list[i].append(value * (10000 if enlarge_values else 1))

    # we default to the median if that does not work
    for x_list, y_list in zip(x_list_list, y_list_list):
        if not is_there_peak(x_list, y_list):
            return x_list, y_list

    return median_aggregate(multiplier_value_to_value_details_dict, enlarge_values)


def most_stable_aggregate_aggregate(multiplier_value_to_value_details_dict, enlarge_values):
    """
    This function aggregates the values from a dictionary of value lists using 
    different aggregation methods (min, max, median) and returns two lists: one for 
    the multiplier values (x_list) and one for the corresponding aggregated values (y_list).
    
    The function first tries to use the min aggregation method. If the resulting 
    aggregated values do not have a peak, it returns the result. Otherwise, it tries 
    the max aggregation method, and if that also results in a peak, it uses the median 
    aggregation method.
    Args:
        multiplier_value_to_value_details_dict (dict): A dictionary where keys are 
            multiplier values and values are dictionaries containing a 'value_list' key.
        enlarge_values (bool): A boolean indicating whether to multiply the aggregated 
            values by 10000.
    Returns:
        tuple: Two lists, x_list and y_list. x_list contains the multiplier values as integers, 
            and y_list contains the corresponding aggregated values.
    Raises:
        Exception: If any value in the 'value_list' is None.
    """

    x_list, y_list = min_aggregate(
        multiplier_value_to_value_details_dict, enlarge_values)

    if not is_there_peak(x_list, y_list):
        return x_list, y_list

    x_list, y_list = max_aggregate(
        multiplier_value_to_value_details_dict, enlarge_values)

    if not is_there_peak(x_list, y_list):
        return x_list, y_list

    x_list, y_list = median_aggregate(
        multiplier_value_to_value_details_dict, enlarge_values)
    return x_list, y_list

def infer_complexity_from_values(
    value_dict,
    complexity_name_function_list,
    filter_outliers=True,
    apply_penalty=True,
    apply_constraints=True,
    zero_out_first_value=True,
    piecewise_fit=True,
    aggregate_y_values='min',
    max_time_rate=0.8,
    elect_complexity='min',
    fix_constant_complexity=True,
    fix_negligeable_complexity=True,
    enlarge_values=True,
    print_info=True,
    multiplier_start=1,
    aggressive_max_time_x_scaling=True,
):
    """
    Infers the complexity of a given piece of code from its runtime or memory footprint values.
    Args:
        value_dict (dict): A dictionary where keys are variable names and values are dictionaries containing 'value_list' key.
        complexity_name_function_list (list): A list of tuples containing complexity name, class, penalty, and order.
        filter_outliers (bool, optional): Whether to filter out outliers in the data. Defaults to True.
        apply_penalty (bool, optional): Whether to apply penalty to the complexity calculation. Defaults to True.
        apply_constraints (bool, optional): Whether to apply constraints to the complexity calculation. Defaults to True.
        zero_out_first_value (bool, optional): Whether to zero out the first value in the data. Defaults to True.
        piecewise_fit (bool, optional): Whether to use piecewise fit for the complexity calculation. Defaults to True.
        aggregate_y_values (str, optional): How to aggregate y-values. Can be 'min', 'max', 'median', 'mean', 'first', 'most_stable_run_aggregate', or 'most_stable_aggregate_aggregate'. Defaults to 'min'.
        max_time_rate (float, optional): The maximum time rate for the complexity calculation. Defaults to 0.8.
        elect_complexity (str, optional): How to elect the complexity. Can be 'min' or 'max'. Defaults to 'min'.
        fix_constant_complexity (bool, optional): Whether to fix constant complexity. Defaults to True.
        fix_negligeable_complexity (bool, optional): Whether to fix negligible complexity. Defaults to True.
        enlarge_values (bool, optional): Whether to enlarge the values. Defaults to True.
        print_info (bool, optional): Whether to print information during the calculation. Defaults to True.
        multiplier_start (int, optional): The starting multiplier for the calculation. Defaults to 1.
        aggressive_max_time_x_scaling (bool, optional): Whether to use aggressive max time x scaling. Defaults to True.
    Returns:
        tuple: A tuple containing the inferred complexity as a string, a boolean indicating whether the complexity has a peak in its curves that were deterministic in the choice of this complexity, a first coefficient that attemps to measure the slope of the curve (experimental, not retained for the benchmark), a second coefficient that is the chosen one as the curve coefficient of the complexity, and finally a boolean about peaks in the measure curves in general
    """

    aggregate_function = None

    if type(aggregate_y_values) == str:
        assert aggregate_y_values in [
            'min_aggregate',
            'max_aggregate',
            'median_aggregate',
            'mean_aggregate',
            'first_aggregate',
            'most_stable_run_aggregate',
            'most_stable_aggregate_aggregate',
        ]

        if aggregate_y_values == 'min_aggregate':
            aggregate_function = min_aggregate

        elif aggregate_y_values == 'max_aggregate':
            aggregate_function = max_aggregate

        elif aggregate_y_values == 'mean_aggregate':
            aggregate_function = mean_aggregate

        elif aggregate_y_values == 'median_aggregate':
            aggregate_function = median_aggregate

        elif aggregate_y_values == 'first_aggregate':
            aggregate_function = first_aggregate

        elif aggregate_y_values == 'most_stable_run_aggregate':
            aggregate_function = most_stable_run_aggregate

        elif aggregate_y_values == 'most_stable_aggregate_aggregate':
            aggregate_function = most_stable_aggregate_aggregate

        else:
            raise Exception('not supported')

    else:
        raise Exception('not supported')

    assert elect_complexity in ['min', 'max']
    elect_function = None

    if elect_complexity == 'min':
        elect_function = np.argmin

    elif elect_complexity == 'max':
        elect_function = np.argmax

    else:
        raise Exception('not supported')

    complexity_details_list = []

    max_time_x_value = list(map(lambda x: max(x) if len(x) > 0 else 0, list(map(lambda multiplier_value_to_value_details_list_dict: [
        int(multiplier_value)
        for multiplier_value, value_details_list in multiplier_value_to_value_details_list_dict.items()
        if len([value_details for value_details in value_details_list if len(value_details['value_list']) > 0]) > 0
    ], value_dict.values()))))

    max_time_x_value = max(max_time_x_value) if len(
        max_time_x_value) > 0 else 0

    if print_info:
        print(max_time_x_value)

    found_peak_max_time_list = []

    for variable_index, (variable_name_type_dimension, multiplier_value_to_value_details_list_dict) in enumerate(value_dict.items()):
        # Now let's explore per run_id, that is to say type of expansions that was tried on the variable
        id_set = set(map(lambda x: x['id_'], [
            value_details
            for value_details_list in list(multiplier_value_to_value_details_list_dict.values())
            for value_details in value_details_list
        ]))

        # variable_name, variable_type, dimension = variable_name_type_dimension
        if print_info:
            print('####################################')
            print(variable_name_type_dimension, 'methods', id_set)

        id_set = sorted(list(id_set))

        for id_ in id_set:
            found_peak = False
            max_time_peak = 0

            if print_info:
                print('############', variable_name_type_dimension, id_)

            multiplier_value_to_value_details_dict = {}

            for multiplier_value, value_details_list in multiplier_value_to_value_details_list_dict.items():
                filtered_value_details_list = list(
                    filter(lambda x: x['id_'] == id_, value_details_list))

                if len(filtered_value_details_list) == 1:
                    multiplier_value_to_value_details_dict[multiplier_value] = filtered_value_details_list[0]

                elif len(filtered_value_details_list) > 1:
                    if print_info:
                        print('weird, skipping...')
                        print(filtered_value_details_list)
                    raise Exception('')

            if len(list(multiplier_value_to_value_details_dict.keys())) <= 3:
                if print_info:
                    print('skipping a method', 1)
                continue

            residuals_list = []
            x_list = []
            y_pulled_list = []
            ref_t_list = []
            t_list = []

            tag_list = None
            priority = None

            for multiplier_value, value_details in multiplier_value_to_value_details_dict.items():
                if any([(value is None) for value in value_details['value_list']]):
                    raise Exception('not supposed to be none')

                if len(value_details['value_list']) == 0:
                    continue

                tag_list = value_details['tag_list']
                priority = value_details['priority']

                break

            x_list, y_pulled_list = aggregate_function(
                multiplier_value_to_value_details_dict, enlarge_values)

            # We filter out values of y that are not increasing but rather decreasing
            x_list_filtered = []
            y_list_filtered = []

            assert len(x_list) == len(y_pulled_list)

            if len(x_list) <= 3:
                if print_info:
                    print('skipping a method', 2)
                continue

            if x_list[0] != multiplier_start:
                if print_info:
                    print('skipping a method', 1)
                continue

            y_pulled_list[0] = min(y_pulled_list)

            for i in range(len(y_pulled_list) - 1):
                if x_list[i] < 256:
                    continue

                if y_pulled_list[i] > y_pulled_list[i+1]:
                    found_peak = True

            if filter_outliers:
                for i in range(0, len(y_pulled_list)):
                    if y_pulled_list[i]/y_pulled_list[-1] < 1.3:
                        x_list_filtered.append(x_list[i])
                        y_list_filtered.append(y_pulled_list[i])

                x_list = x_list_filtered[:]
                y_pulled_list = y_list_filtered[:]

            x_list_filtered = []
            y_list_filtered = []

            assert len(x_list) == len(y_pulled_list)

            if len(x_list) <= 3:
                if print_info:
                    print('skipping a method', 3)
                continue

            if filter_outliers:
                # we do not handle small values of x

                x_start = 512
                x_list_filtered.append(x_list[0])
                y_list_filtered.append(y_pulled_list[0])
                offset = 0

                for i in range(1, len(y_pulled_list)-1):
                    if x_list[i] <= x_start:
                        x_list_filtered.append(x_list[i])
                        y_list_filtered.append(y_pulled_list[i])

                    else:
                        # we distinguish the case of a peak, and the case where there is no peak
                        if y_pulled_list[i-1] <= y_pulled_list[i]:
                            x_list_filtered.append(x_list[i])
                            y_list_filtered.append(y_pulled_list[i] + offset)

                        else:
                            # we look at the value after to distinguish temp peak and permanent peak
                            if y_pulled_list[i-1] < y_pulled_list[i+1]:
                                assert y_pulled_list[i] < y_pulled_list[i+1]
                                # it is a temp peak
                                # we just skip it
                                continue

                            elif y_pulled_list[i] < y_pulled_list[i+1]:
                                #  but therefore y_pulled_list[i-1] > y_pulled_list[i+1]
                                #  it is a permanent peak
                                break
                                offset += (y_pulled_list[i-1] - y_pulled_list[i]) +\
                                    (x_list[i] - x_list[i-1]) * (y_pulled_list[i-1] -
                                                                 y_pulled_list[i-2])/(x_list[i-1] - x_list[i-2])
                                x_list_filtered.append(x_list[i])
                                y_list_filtered.append(
                                    y_pulled_list[i] + offset)

                            else:
                                #  it keeps falling, we just add the values
                                x_list_filtered.append(x_list[i])
                                y_list_filtered.append(
                                    y_pulled_list[i] + offset)

                # case to add the last value
                else:
                    if y_pulled_list[-2] <= y_pulled_list[-1]:
                        x_list_filtered.append(x_list[-1])
                        y_list_filtered.append(y_pulled_list[-1] + offset)

            else:
                x_list_filtered = x_list
                y_list_filtered = y_pulled_list

            assert len(x_list_filtered) == len(y_list_filtered)
            if len(x_list_filtered) <= 2:
                if print_info:
                    print('skipping a method', 4)
                continue

            max_time_list = []
            coeff_list = []

            max_time_peak = y_list_filtered[-1]

            for complexity_name, complexity_class, penalty, order_ in complexity_name_function_list:
                complexity_instance = complexity_class()
                residuals, ref_t, t, max_time_temp, coeff = complexity_instance.fit(
                    x_list_filtered,
                    y_list_filtered,
                    apply_constraints=apply_constraints,
                    zero_out_first_value=zero_out_first_value,
                    piecewise_fit=piecewise_fit,
                    max_time_x_value=max_time_x_value if aggressive_max_time_x_scaling else min(
                        max_time_x_value, max(x_list_filtered)),
                )

                assert len(x_list_filtered) == len(ref_t)
                assert len(x_list_filtered) == len(t)

                max_time_list.append(max_time_temp)
                coeff_list.append(coeff)

                residuals_list.append(
                    (residuals + 0) * (penalty if apply_penalty else 1)
                )

                ref_t_list.append(ref_t)
                t_list.append(t)

            index_of_constant_complexity = list(map(lambda x: x[0], filter(
                lambda x: x[1] == 'o(1)',
                enumerate(map(lambda x: x[0], complexity_name_function_list))
            )))[0]

            if math.isclose(residuals_list[index_of_constant_complexity], 0, rel_tol=1e-8):
                residuals_list[index_of_constant_complexity] = 0

                for i in range(0, len(residuals_list)):
                    if i == index_of_constant_complexity:
                        continue

                    residuals_list[i] = max(residuals_list[i], 1e-5)

            complexity_details_list.append(
                {
                    'variable_name_type_dimension': variable_name_type_dimension,
                    'class_': complexity_name_function_list[np.argmin(residuals_list)][0],
                    'order': complexity_name_function_list[np.argmin(residuals_list)][3],
                    'class_backup': complexity_name_function_list[[i for i in np.argsort(residuals_list) if complexity_name_function_list[i][0] != 'o(1)'][0]][0],
                    'order_backup': complexity_name_function_list[[i for i in np.argsort(residuals_list) if complexity_name_function_list[i][0] != 'o(1)'][0]][3],
                    'found_peak': found_peak,
                    'max_time': max_time_list[np.argmin(residuals_list)],
                    'priority': priority,
                    'tag_list': tag_list,
                    'number_variables': len(tag_list),
                    'variable_index': variable_index,
                    'coeff':  coeff_list[np.argmin(residuals_list)],
                    'coeff_backup':  coeff_list[[i for i in np.argsort(residuals_list) if complexity_name_function_list[i][0] != 'o(1)'][0]],
                }
            )

            found_peak_max_time_list.append((found_peak, max_time_peak))

            if print_info:
                print(
                    'complexity', complexity_name_function_list[np.argmin(residuals_list)][0])
                print('time', max_time_list[np.argmin(residuals_list)])
                print('coeff', coeff_list[np.argmin(residuals_list)])
                print('tag_list', tag_list)

            if print_info:
                # # if variable_index in [0, 1, 2, 3, 4, 5]:
                # if id_ != 'copy_other_large':
                #     continue
                print(residuals_list)

                print(1)
                p = plt.plot(
                    x_list_filtered,
                    ref_t_list[np.argmin(residuals_list)],
                    '--'
                )
                p = plt.plot(
                    x_list_filtered,
                    t,
                    color=p[0].get_color(),
                    label=str(variable_index) + ', ' + id_
                )

    # now we need to handle this list
    complexity_details_list.sort(key=lambda x: (
        x['number_variables'], x['variable_index']))
    variable_name_type_dimension_to_complexity_group_dict = {}

    for variable_name_type_dimension in value_dict.keys():
        # variable_name, variable_type, dimension = variable_name_type_dimension
        node = NodeComplexity(variable_name_type_dimension)
        node.print_info = print_info
        variable_name_type_dimension_to_complexity_group_dict[
            variable_name_type_dimension
        ] = node

    if print_info:
        print(complexity_details_list)

    for complexity_details in complexity_details_list:
        if complexity_details['number_variables'] == 1:
            variable_name_type_dimension_to_complexity_group_dict[
                complexity_details['variable_name_type_dimension']
            ].add_complexity(ComplexityCandidate(
                class_=complexity_details['class_'],
                order=complexity_details['order'],
                class_backup=complexity_details['class_backup'],
                order_backup=complexity_details['order_backup'],
                max_time=complexity_details['max_time'],
                priority=complexity_details['priority'],
                found_peak=complexity_details['found_peak'],
                coeff=complexity_details['coeff'],
                coeff_backup=complexity_details['coeff_backup'],
            ))

        else:
            assert complexity_details['number_variables'] > 1

            # either they are all in the same group, either we add a new super group
            complexity_group_list = [
                variable_name_type_dimension_to_complexity_group_dict[
                    variable_name_type_dimension
                ] for variable_name_type_dimension in complexity_details['tag_list']
            ]

            if len(set(complexity_group_list)) == 1:
                complexity_group_list[0].add_complexity(ComplexityCandidate(
                    class_=complexity_details['class_'],
                    order=complexity_details['order'],
                    class_backup=complexity_details['class_backup'],
                    order_backup=complexity_details['order_backup'],
                    max_time=complexity_details['max_time'],
                    priority=complexity_details['priority'],
                    found_peak=complexity_details['found_peak'],
                    coeff=complexity_details['coeff'],
                    coeff_backup=complexity_details['coeff_backup'],
                ))

            else:
                group_complexity = GroupComplexity()
                group_complexity.print_info = print_info

                for group_or_node in set(complexity_group_list):
                    group_complexity.add_group_or_node(group_or_node)

                group_complexity.add_complexity(ComplexityCandidate(
                    class_=complexity_details['class_'],
                    order=complexity_details['order'],
                    class_backup=complexity_details['class_backup'],
                    order_backup=complexity_details['order_backup'],
                    max_time=complexity_details['max_time'],
                    priority=complexity_details['priority'],
                    found_peak=complexity_details['found_peak'],
                    coeff=complexity_details['coeff'],
                    coeff_backup=complexity_details['coeff_backup'],
                ))

                for variable_name_type_dimension in complexity_details['tag_list']:
                    variable_name_type_dimension_to_complexity_group_dict[
                        variable_name_type_dimension
                    ] = group_complexity

                for variable_name_type_dimension, complexity_group in variable_name_type_dimension_to_complexity_group_dict.items():
                    if complexity_group in set(complexity_group_list):
                        variable_name_type_dimension_to_complexity_group_dict[
                            variable_name_type_dimension
                        ] = group_complexity

    # At least one variable must have some complexity candidates, or be part of a group with complexity candidates
    if not any([
        len(complexity_group.complexity_candidate_list) > 0 for complexity_group in variable_name_type_dimension_to_complexity_group_dict.values()
    ]):
        return None, False, None, None, None

    if len(variable_name_type_dimension_to_complexity_group_dict.values()) == 0:
        return None, False, None, None, None

    complexity_group_list = [
        x for x in set(variable_name_type_dimension_to_complexity_group_dict.values())
    ]

    if print_info:
        print('number of resulting groups', len(complexity_group_list))

    main_complexity_group = None

    if len(set(complexity_group_list)) == 1:
        main_complexity_group = complexity_group_list[0]

    else:
        main_complexity_group = GroupComplexity()
        main_complexity_group.print_info = print_info

        for group_or_node in set(complexity_group_list):
            main_complexity_group.add_group_or_node(group_or_node)

    if print_info:
        print('below the tree')
        print(main_complexity_group.get_encapsulated_variable_name_type_dimension())

    main_complexity_group.self_set_complexity(
        max_time=None,
        fix_negligeable_complexity=fix_negligeable_complexity,
        max_time_rate=max_time_rate,
        elect_function=elect_function,
        fix_constant_complexity=fix_constant_complexity,
    )

    main_complexity_group.self_adjust_group_operations(
        max_time=None,
        fix_negligeable_complexity=fix_negligeable_complexity,
        max_time_rate=max_time_rate,
        elect_function=elect_function,
        fix_constant_complexity=fix_constant_complexity,
    )

    main_complexity_group.self_adjust_constant_complexity(
        max_time=None,
        fix_negligeable_complexity=fix_negligeable_complexity,
        max_time_rate=max_time_rate,
        elect_function=elect_function,
        fix_constant_complexity=fix_constant_complexity,
    )

    main_complexity_group.self_adjust_group_operations(
        max_time=None,
        fix_negligeable_complexity=fix_negligeable_complexity,
        max_time_rate=max_time_rate,
        elect_function=elect_function,
        fix_constant_complexity=fix_constant_complexity,
    )

    if print_info:
        plt.legend()

    formatted_complexity = main_complexity_group.format_complexity(
        letter_list=['n', 'm', 'k', 'l', 'u', 'v', 'w'],
        next_letter_index=0,
    )

    return 'o({})'.format(formatted_complexity[0]), False, formatted_complexity[2], formatted_complexity[3], bool(
        sum(list(map(lambda x: int(x[0]), list(filter(lambda x: x[1] >= max(list(map(
            lambda x: x[1], found_peak_max_time_list))) * 0.8, found_peak_max_time_list))))) > 0
    ) if len(found_peak_max_time_list) else None
