# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def correct_complexity_formatting(complexity):
    if complexity is None:
        return None
    
    if complexity[0] == "o" or complexity[0] == "O":
        complexity = complexity[1:]

    while len(complexity) >= 2 and complexity[0] == "(" and complexity[-1] == ")":
        complexity = complexity[1:-1]

    return f"O({complexity})"


def convert_measures_dict_format(
    measures_dict
):
    converted_measures_list = []

    for input_id in measures_dict.keys():
        measures_per_multiplier = []

        for multiplier, value_list in measures_dict[input_id].items():
            measures_per_method = []

            for value_ in value_list:
                measures_per_method.append(
                    {
                        "value_list": value_["value_list"],
                        "expansion_method": value_["id_"],
                        "measures_set_id_list": [str(x) for x in value_["tag_list"]],
                        "measures_priority": value_["priority"],
                    }
                )

            measures_per_multiplier.append(
                {
                    "expansion_multiplier": int(multiplier),
                    "measures_per_expansion_method": measures_per_method,
                }
            )

        converted_measures_list.append(
            {
                "measures_set_id": input_id,
                "measures_per_expansion_multiplier": measures_per_multiplier
            }
        )

    return converted_measures_list


def convert_measures_set_id_to_input_properties_format(
    index_variable_dict
):
    measures_set_index_to_input_properties = dict()

    for key_, value_ in index_variable_dict.items():
        measures_set_index_to_input_properties[str(value_)] = {
            "input_id": key_.split("####")[0],
            "framework_input_type": key_.split("####")[1],
            "input_dimension": (
                key_.split("####")[2] if key_.split("####")[
                    2] not in ['', "None", "none"] else None
            ),
        }

    return measures_set_index_to_input_properties
