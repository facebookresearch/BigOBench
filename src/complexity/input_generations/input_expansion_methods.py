# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faulthandler
import json
import logging
import signal
import sys
import resource
import traceback
from io import StringIO
from multiprocessing.connection import Connection
from typing import Any, Dict
import builtins
from unittest.mock import mock_open
import json
import re
import linecache
import subprocess
import sys
import tempfile
import signal
import time
from multiprocessing.connection import Connection
from string import Template
import string
import time
import cProfile
import random

from .atomic_types import (
    string_multiplier_identity, 
    string_base_identity, 
    string_base_copy, 
    string_base_random, 
    string_multiplier_random, 
    int_multiplier, 
    int_base_copy, 
    float_multiplier, 
    float_base_copy,
)

from .one_dim_list_types import (
    list_multiplier_identity, 
    list_string_base_identity, 
    list_string_base_copy, 
    list_string_base_random, 
    list_string_multiplier_random, 
    list_int_base_identity, 
    list_int_base_random, 
    list_int_base_random_verylarge, 
    list_int_base_copy, 
    list_int_multiplier_random, 
    list_int_multiplier_random_verylarge, 
    list_float_base_identity, 
    list_float_base_random, 
    list_float_base_copy, 
    list_float_multiplier_random, 
    list_bool_base_identity, 
    list_bool_base_random, 
    list_bool_base_copy, 
    list_bool_multiplier_random,
)

from .tuple_list_types import (
    list_tuple_string_base_identity, 
    list_tuple_string_base_copy, 
    list_tuple_string_base_random, 
    list_tuple_string_multiplier_random, 
    list_tuple_int_base_identity, 
    list_tuple_int_base_random, 
    list_tuple_int_base_random_verylarge, 
    list_tuple_int_base_copy, 
    list_tuple_int_multiplier_random, 
    list_tuple_int_multiplier_random_verylarge, 
    list_tuple_float_base_identity, 
    list_tuple_float_base_random, 
    list_tuple_float_base_copy, 
    list_tuple_float_multiplier_random, 
    list_tuple_bool_base_identity, 
    list_tuple_bool_base_random, 
    list_tuple_bool_base_copy, 
    list_tuple_bool_multiplier_random,
)

from .two_dim_list_types import (
    list_list_multiplier_identity, 
    list_list_string_base_identity, 
    list_list_string_base_random, 
    list_list_string_base_copy, 
    list_list_string_multiplier_random_dim2, 
    list_list_int_base_identity, 
    list_list_int_base_random, 
    list_list_int_base_copy, 
    list_list_int_multiplier_random_dim2, 
    list_list_float_base_identity, 
    list_list_float_base_random, 
    list_list_float_base_copy, 
    list_list_float_multiplier_random_dim2, 
    list_list_bool_base_identity, 
    list_list_bool_base_random, 
    list_list_bool_base_copy, 
    list_list_bool_multiplier_random_dim2, 
    list_list_string_multiplier_random_dim1, 
    list_list_int_multiplier_random_dim1, 
    list_list_float_multiplier_random_dim1, 
    list_list_bool_multiplier_random_dim1,
)

def get_expansion_details():
    # Defines a dict of the following type that records all ways to increase in size a specific data type
    # variable_type_dimension_method_to_base_multiplier_priority = {
    #     (variable_type, dimension): {
    #         method_1: {
    #             'base': base_function, 
    #             'multiplier': multiplier_function, 
    #             'priority': priority_score
    #         },
    #         method_2: {
    #             'base': base_function, 
    #             'multiplier': multiplier_function, 
    #             'priority': priority_score
    #         },
    #     },
    # }
    variable_type_dimension_method_to_base_multiplier_priority = {
        ("<class 'str'>\n", None): {
            'copy': {
                'base': string_base_copy, 
                'multiplier': string_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': string_base_identity, 
                'multiplier': string_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': string_base_random, 
                'multiplier': string_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'int'>\n", None): {
            'copy': {
                'base': int_base_copy, 
                'multiplier': int_multiplier, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'float'>\n", None): {
            'copy': {
                'base': float_base_copy, 
                'multiplier': float_multiplier, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },



        # "<class 'list'>\n": None,
        ("<class 'list'>\n<class 'str'>\n", 1): {
            'copy': {
                'base': list_string_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_string_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_string_base_random, 
                'multiplier': list_string_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'int'>\n", 1): {
            'copy': {
                'base': list_int_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_int_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_int_base_random, 
                'multiplier': list_int_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_verylarge': {
                'base': list_int_base_random_verylarge, 
                'multiplier': list_int_multiplier_random_verylarge, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'float'>\n", 1): {
            'copy': {
                'base': list_float_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_float_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_float_base_random, 
                'multiplier': list_float_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'bool'>\n", 1): {
            'copy': {
                'base': list_bool_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_bool_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_bool_base_random, 
                'multiplier': list_bool_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },



        # "<class 'list'>\n<class 'tuple'>\n": None,
        ("<class 'list'>\n<class 'tuple'>\n<class 'str'>\n", 1): {
            'copy': {
                'base': list_tuple_string_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_tuple_string_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_tuple_string_base_random, 
                'multiplier': list_tuple_string_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'tuple'>\n<class 'int'>\n", 1): {
            'copy': {
                'base': list_tuple_int_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_tuple_int_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_tuple_int_base_random, 
                'multiplier': list_tuple_int_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_verylarge': {
                'base': list_tuple_int_base_random_verylarge, 
                'multiplier': list_tuple_int_multiplier_random_verylarge, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'tuple'>\n<class 'float'>\n", 1): {
            'copy': {
                'base': list_tuple_float_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_tuple_float_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_tuple_float_base_random, 
                'multiplier': list_tuple_float_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'tuple'>\n<class 'bool'>\n", 1): {
            'copy': {
                'base': list_tuple_bool_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_tuple_bool_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_tuple_bool_base_random, 
                'multiplier': list_tuple_bool_multiplier_random, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        


        # "<class 'list'>\n<class 'list'>\n": None,
        ("<class 'list'>\n<class 'list'>\n<class 'str'>\n", 1): {
            'copy': {
                'base': list_list_string_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_list_string_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_list_string_base_random, 
                'multiplier': list_list_string_multiplier_random_dim1, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'copy_2dim': {
                'base': list_list_string_base_copy,
                'multiplier': lambda x, n: list_multiplier_identity(list_list_multiplier_identity(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_2dim': {
                'base': list_list_string_base_random,
                'multiplier': lambda x, n: list_list_string_multiplier_random_dim1(list_list_string_multiplier_random_dim2(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'list'>\n<class 'int'>\n", 1): {
            'copy': {
                'base': list_list_int_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_list_int_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_list_int_base_random, 
                'multiplier': list_list_int_multiplier_random_dim1, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'copy_2dim': {
                'base': list_list_int_base_copy,
                'multiplier': lambda x, n: list_multiplier_identity(list_list_multiplier_identity(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_2dim': {
                'base': list_list_int_base_random,
                'multiplier': lambda x, n: list_list_int_multiplier_random_dim1(list_list_int_multiplier_random_dim2(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'list'>\n<class 'float'>\n", 1): {
            'copy': {
                'base': list_list_float_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_list_float_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_list_float_base_random, 
                'multiplier': list_list_float_multiplier_random_dim1, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'copy_2dim': {
                'base': list_list_float_base_copy,
                'multiplier': lambda x, n: list_multiplier_identity(list_list_multiplier_identity(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_2dim': {
                'base': list_list_float_base_random,
                'multiplier': lambda x, n: list_list_float_multiplier_random_dim1(list_list_float_multiplier_random_dim2(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'list'>\n<class 'bool'>\n", 1): {
            'copy': {
                'base': list_list_bool_base_copy, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_list_bool_base_identity, 
                'multiplier': list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_list_bool_base_random, 
                'multiplier': list_list_bool_multiplier_random_dim1, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'copy_2dim': {
                'base': list_list_bool_base_copy,
                'multiplier': lambda x, n: list_multiplier_identity(list_list_multiplier_identity(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_2dim': {
                'base': list_list_bool_base_random,
                'multiplier': lambda x, n: list_list_bool_multiplier_random_dim1(list_list_bool_multiplier_random_dim2(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'list'>\n<class 'str'>\n", 2): {
            'copy': {
                'base': list_list_string_base_copy, 
                'multiplier': list_list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_list_string_base_identity, 
                'multiplier': list_list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_list_string_base_random, 
                'multiplier': list_list_string_multiplier_random_dim2, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'copy_2dim': {
                'base': list_list_string_base_copy,
                'multiplier': lambda x, n: list_multiplier_identity(list_list_multiplier_identity(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_2dim': {
                'base': list_list_string_base_random,
                'multiplier': lambda x, n: list_list_string_multiplier_random_dim1(list_list_string_multiplier_random_dim2(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'list'>\n<class 'int'>\n", 2): {
            'copy': {
                'base': list_list_int_base_copy, 
                'multiplier': list_list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_list_int_base_identity, 
                'multiplier': list_list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_list_int_base_random, 
                'multiplier': list_list_int_multiplier_random_dim2, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'copy_2dim': {
                'base': list_list_int_base_copy,
                'multiplier': lambda x, n: list_multiplier_identity(list_list_multiplier_identity(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_2dim': {
                'base': list_list_int_base_random,
                'multiplier': lambda x, n: list_list_int_multiplier_random_dim1(list_list_int_multiplier_random_dim2(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'list'>\n<class 'float'>\n", 2): {
            'copy': {
                'base': list_list_float_base_copy, 
                'multiplier': list_list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_list_float_base_identity, 
                'multiplier': list_list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_list_float_base_random, 
                'multiplier': list_list_float_multiplier_random_dim2, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'copy_2dim': {
                'base': list_list_float_base_copy,
                'multiplier': lambda x, n: list_multiplier_identity(list_list_multiplier_identity(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_2dim': {
                'base': list_list_float_base_random,
                'multiplier': lambda x, n: list_list_float_multiplier_random_dim1(list_list_float_multiplier_random_dim2(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
        ("<class 'list'>\n<class 'list'>\n<class 'bool'>\n", 2): {
            'copy': {
                'base': list_list_bool_base_copy, 
                'multiplier': list_list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'identity': {
                'base': list_list_bool_base_identity, 
                'multiplier': list_list_multiplier_identity, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random': {
                'base': list_list_bool_base_random, 
                'multiplier': list_list_bool_multiplier_random_dim2, 
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'copy_2dim': {
                'base': list_list_bool_base_copy,
                'multiplier': lambda x, n: list_multiplier_identity(list_list_multiplier_identity(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
            'random_2dim': {
                'base': list_list_bool_base_random,
                'multiplier': lambda x, n: list_list_bool_multiplier_random_dim1(list_list_bool_multiplier_random_dim2(x, n, print_=False), n),
                'priority': 1,
                'used_for_base_input_precomputation': True,
            },
        },
    }   

    # Compute all the variables type dimension that exist

    variable_type_dimension_set = set()
    for variable_type_dimension in variable_type_dimension_method_to_base_multiplier_priority.keys():
        variable_type_dimension_set.add(variable_type_dimension)

    # Let's generate the following one a bit automatically, using variable_type_dimension_method_to_base_multiplier_priority
    # And record an example of course

    variable_type_dimension_to_expansion_details_list = {
    }

    for variable_type_dimension in variable_type_dimension_set:

        # Let's add the classic method with constant value of the other argumentsx
        for method in {'copy', 'identity', 'random', 'random_verylarge'}:
            if method in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():

                # if the variable_type_dimension is new
                if variable_type_dimension not in variable_type_dimension_to_expansion_details_list.keys():
                    variable_type_dimension_to_expansion_details_list[variable_type_dimension] = []

                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            method, 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension][method]['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension][method]['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (name_ref != name_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                        ],
                        'tag': (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (name_ref == name_targ) and (type_ref == type_targ) and (dim_ref == dim_targ)
                            )
                        ),
                        'tag_min_count': 1,
                        'id_': method + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension][method]['priority'],
                    }
                )

                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            method, 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension][method]['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension][method]['multiplier'],
                        'target_filter_base_multiplier_list':[
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (name_ref != name_targ)),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': None,
                            },
                        ],
                        'tag': (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (name_ref == name_targ) and (type_ref == type_targ) and (dim_ref == dim_targ)
                            )
                        ),
                        'tag_min_count': 1,
                        'id_': method + '_other_small',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension][method]['priority'],# + 1,
                    }
                )

        if (variable_type_dimension[0] in [
            "<class 'list'>\n<class 'list'>\n<class 'str'>\n",
            "<class 'list'>\n<class 'list'>\n<class 'int'>\n",
            "<class 'list'>\n<class 'list'>\n<class 'float'>\n",
            "<class 'list'>\n<class 'list'>\n<class 'bool'>\n",
        ]) and (variable_type_dimension[1] == 2):

            # in this case we also examine the covariance of the dimensions of the array
            if 'copy_2dim' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'copy_2dim', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy_2dim']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy_2dim']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (name_ref != name_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (name_ref == name_targ) and (type_ref == type_targ) and (dim_targ in [1, 2])
                            )
                        ),
                        'tag_min_count': 2,
                        'id_': 'copy_2dim' + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy_2dim']['priority'],
                    }
                )

            if 'random_2dim' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'random_2dim', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random_2dim']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random_2dim']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (name_ref != name_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (name_ref == name_targ) and (type_ref == type_targ) and (dim_targ in [1, 2])
                            )
                        ),
                        'tag_min_count': 2,
                        'id_': 'random_2dim' + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random_2dim']['priority'],
                    }
                )

        if (variable_type_dimension[0] in [
            "<class 'list'>\n<class 'int'>\n",
            "<class 'list'>\n<class 'str'>\n",
            "<class 'list'>\n<class 'float'>\n",
            "<class 'list'>\n<class 'bool'>\n",
        ]) and (variable_type_dimension[1] == 1):

            # we make two lists vary together !
            if 'random' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'random', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref != type_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ)),
                                'target_base': ('random', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (type_ref == type_targ)
                            )
                        ),
                        'tag_min_count': 2,
                        'id_': 'random_all_lists' + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['priority'],
                    }
                )

            # we make a list covary with nearby ints !
            if 'random' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'random', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ != "<class 'int'>\n") and (type_ref != type_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ)),
                                'target_base': ('random', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ == "<class 'int'>\n")),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[("<class 'int'>\n", None)]['copy']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                ((name_targ, type_targ, dim_targ) == (name_ref, type_ref, dim_ref)) or (type_targ == "<class 'int'>\n") or (type_ref == type_targ)
                            )
                        ),
                        'id_': 'random_list_and_all_ints' + '_other_large',
                        'tag_min_count': 2,
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['priority'],
                    }
                )

            if 'copy' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'copy', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ != "<class 'int'>\n") and (type_ref != type_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ)),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ == "<class 'int'>\n")),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[("<class 'int'>\n", None)]['copy']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                ((name_targ, type_targ, dim_targ) == (name_ref, type_ref, dim_ref)) or (type_targ == "<class 'int'>\n") or (type_ref == type_targ)
                            )
                        ),
                        'id_': 'copy_list_and_all_ints' + '_other_large',
                        'tag_min_count': 2,
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['priority'],
                    }
                )

        # Specific to code contest
        if (variable_type_dimension[0] in [
            "<class 'list'>\n<class 'list'>\n<class 'int'>\n",
            "<class 'list'>\n<class 'list'>\n<class 'str'>\n",
            "<class 'list'>\n<class 'list'>\n<class 'float'>\n",
            "<class 'list'>\n<class 'list'>\n<class 'bool'>\n",
            "<class 'list'>\n<class 'tuple'>\n<class 'int'>\n",
            "<class 'list'>\n<class 'tuple'>\n<class 'str'>\n",
            "<class 'list'>\n<class 'tuple'>\n<class 'float'>\n",
            "<class 'list'>\n<class 'tuple'>\n<class 'bool'>\n",

        ]) and (variable_type_dimension[1] == 1):
            # we make a list covary with nearby ints !
            if 'random' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'random', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ != "<class 'int'>\n") and (type_ref != type_targ or dim_ref != dim_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ) and (dim_ref == dim_targ)),
                                'target_base': ('random', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ == "<class 'int'>\n")),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[("<class 'int'>\n", None)]['copy']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                ((name_targ, type_targ, dim_targ) == (name_ref, type_ref, dim_ref)) or (type_targ == "<class 'int'>\n") or (type_ref == type_targ and dim_ref == dim_targ)
                            )
                        ),
                        'id_': 'random_2dim_list_and_all_ints' + '_other_large',
                        'tag_min_count': 2,
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['priority'],
                    }
                )
            if 'copy' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'copy', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ != "<class 'int'>\n") and (type_ref != type_targ or dim_ref != dim_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ) and (dim_ref == dim_targ)),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ == "<class 'int'>\n")),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[("<class 'int'>\n", None)]['copy']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                ((name_targ, type_targ, dim_targ) == (name_ref, type_ref, dim_ref)) or (type_targ == "<class 'int'>\n") or (type_ref == type_targ and dim_ref == dim_targ)
                            )
                        ),
                        'id_': 'copy_2dim_list_and_all_ints' + '_other_large',
                        'tag_min_count': 2,
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['priority'],
                    }
                )

        if (variable_type_dimension[0] in [
            "<class 'str'>\n",
        ]) and (variable_type_dimension[1] == None):

            # we make two str vary together !
            if 'random' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'random', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref != type_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ)),
                                'target_base': ('random', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (type_ref == type_targ)
                            )
                        ),
                        'tag_min_count': 2,
                        'id_': 'random_all_string' + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['priority'],
                    }
                )

            if 'copy' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'copy', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref != type_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ)),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (type_ref == type_targ)
                            )
                        ),
                        'tag_min_count': 2,
                        'id_': 'copy_all_string' + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['priority'],
                    }
                )

            # and now let's compare with all strings AND ints at the same time !
            if 'random' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'random', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref != type_targ) and (type_targ != "<class 'int'>\n")),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ == "<class 'int'>\n")),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[("<class 'int'>\n", None)]['copy']['multiplier'],
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ)),
                                'target_base': ('random', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (type_ref == type_targ) or (type_targ == "<class 'int'>\n")
                            )
                        ),
                        'tag_min_count': 2,
                        'id_': 'random_all_string_and_all_ints' + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['random']['priority'],
                    }
                )

            if 'copy' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'copy', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref != type_targ) and (type_targ != "<class 'int'>\n")),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_targ == "<class 'int'>\n")),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[("<class 'int'>\n", None)]['copy']['multiplier'],
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ)),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (type_ref == type_targ) or (type_targ == "<class 'int'>\n")
                            )
                        ),
                        'tag_min_count': 2,
                        'id_': 'copy_all_string_and_all_ints' + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['priority'],
                    }
                )


        if (variable_type_dimension[0] in [
            "<class 'int'>\n",
        ]) and (variable_type_dimension[1] == None):
            if 'copy' in variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension].keys():
                variable_type_dimension_to_expansion_details_list[variable_type_dimension].append(
                    {
                        'ref_base': (
                            'copy', 
                            'small', 
                            variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['base']
                        ),
                        'ref_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                        'target_filter_base_multiplier_list': [
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref != type_targ)),
                                'target_base': ('copy', 'large', None),
                                'target_multiplier': None,
                            },
                            {
                                'target_filter': (lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (type_ref == type_targ)),
                                'target_base': ('copy', 'small', None),
                                'target_multiplier': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['multiplier'],
                            },
                        ],
                        'tag':  (
                            lambda name_ref, type_ref, dim_ref, name_targ, type_targ, dim_targ: (
                                (type_ref == type_targ)
                            )
                        ),
                        'tag_min_count': 2,
                        'id_': 'copy_all_int' + '_other_large',
                        'priority': variable_type_dimension_method_to_base_multiplier_priority[variable_type_dimension]['copy']['priority'],
                    }
                )

    return variable_type_dimension_method_to_base_multiplier_priority, variable_type_dimension_to_expansion_details_list