# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

############################################## SINGLETON TYPES ##############################################

def string_multiplier_identity(input_string, n):
    return 'print(' + input_string + '*{})'.format(n)

def string_base_identity(input_string):
    return 'print("a" * len(' + input_string + '))'

def string_base_copy(input_string):
    return 'print(' + input_string + ')'

def string_base_random(input_string):
    return 'print("".join([random.choice(string.ascii_letters.lower()) for _ in range(len(' + input_string + '))]))'

def string_multiplier_random(input_string, n):
    n -= 1
    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print(' 
        + 
        '"".join([random.choice(string.ascii_letters.lower()) for _ in range(len('
        +
        input_string
        +
        ') * {})]) + '.format(n_1)
        +
        input_string
        +
        ' + "".join([random.choice(string.ascii_letters.lower()) for _ in range(len('
        +
        input_string
        +
        ') * {})])'.format(n_2)
        +
        ')'
    )

def int_multiplier(input_int, n):
    return 'print(' + input_int + '*{})'.format(n)

def int_base_copy(input_int):
    return 'print(' + input_int + ')'

def float_multiplier(input_float, n):
    return 'print(' + input_float + '*{})'.format(n)

def float_base_copy(input_float):
    return 'print(' + input_float + ')'