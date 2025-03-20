# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

############################################## 1D LIST-TUPLE TYPES ##############################################

def list_tuple_string_base_identity(input_list):
    return (
        'print([tuple("aaaaaa" for _ in range(len(' + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )

def list_tuple_string_base_copy(input_list):
    return 'print(' + input_list + ')'

def list_tuple_string_base_random(input_list):
    return (
        'print([tuple("".join([random.choice(string.ascii_letters.lower()) for _ in range(random.randint(3, 10))]) for _ in range(len('
        + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )

def list_tuple_string_multiplier_random(input_list, n):
    n -= 1
    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print([tuple("".join([random.choice(string.ascii_letters.lower()) for _ in range(random.randint(3, {}))]) for _ in range(len('.format(max(n, 3)) 
        + input_list +'[0]))) for x in ['
        + 
        input_list 
        + 
        'for _ in range({})] for y in x] + '.format(n_1)
        + 
        input_list
        +
        ' + [tuple("".join([random.choice(string.ascii_letters.lower()) for _ in range(random.randint(3, {}))]) for _ in range(len('.format(max(n, 3))
        + input_list +'[0]))) for x in ['
        +
        input_list
        +
        'for _ in range({})] for y in x])'.format(n_2)
    )

def list_tuple_int_base_identity(input_list):
    return (
        'print([tuple(9998 for _ in range(len(' + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )

def list_tuple_int_base_random(input_list):
    return (
        'print([tuple(random.randint(1, 1000) for _ in range(len(' + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )

def list_tuple_int_base_random_verylarge(input_list):
    return (
        'print([tuple(random.randint(1000000, 1000000 * 10) for _ in range(len(' + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )


def list_tuple_int_base_copy(input_list):
    return (
        'print('
        +
        input_list
        +
        ')'
    )

def list_tuple_int_multiplier_random(input_list, n):
    n -= 1
    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print([tuple(random.randint(1, {}) for _ in range(len('.format(max(n, 1))
        + input_list +'[0]))) for x in ['
        + 
        input_list 
        + 
        'for _ in range({})] for y in x] + '.format(n_1)
        + 
        input_list
        +
        ' + [tuple(random.randint(1, {}) for _ in range(len('.format(max(n, 1))
        + input_list +'[0]))) for x in ['
        +
        input_list
        +
        'for _ in range({})] for y in x])'.format(n_2)
    )

def list_tuple_int_multiplier_random_verylarge(input_list, n):
    n -= 1
    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print([tuple(random.randint(1000000, {}) for _ in range(len('.format(max(n, 1000000 * 100))
        + input_list +'[0]))) for x in ['
        + 
        input_list 
        + 
        'for _ in range({})] for y in x] + '.format(n_1)
        + 
        input_list
        +
        ' + [tuple(random.randint(1000000, {}) for _ in range(len('.format(max(n, 1000000 * 100))
        + input_list +'[0])))  for x in ['
        +
        input_list
        +
        'for _ in range({})] for y in x])'.format(n_2)
    )

def list_tuple_float_base_identity(input_list):
    return (
        'print([tuple(9998.234426346 for _ in range(len(' + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )

def list_tuple_float_base_random(input_list):
    return (
        'print([tuple(random.uniform(0, 1000) for _ in range(len(' + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )

def list_tuple_float_base_copy(input_list):
    return (
        'print('
        +
        input_list
        +
        ')'
    )

def list_tuple_float_multiplier_random(input_list, n):
    n -= 1
    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print([tuple(random.uniform(0, {}) for _ in range(len('.format(n)
        + input_list +'[0]))) for x in ['
        + 
        input_list 
        + 
        'for _ in range({})] for y in x] + '.format(n_1)
        + 
        input_list
        +
        ' + [tuple(random.uniform(0, {}) for _ in range(len('.format(n)
        + input_list +'[0])))  for x in ['
        +
        input_list
        +
        'for _ in range({})] for y in x])'.format(n_2)
    )

def list_tuple_bool_base_identity(input_list):
    return (
        'print([tuple(True for _ in range(len(' + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )

def list_tuple_bool_base_random(input_list):
    return (
        'print([tuple(bool(random.getrandbits(1)) for _ in range(len(' + input_list +'[0]))) for _ in '
        +
        input_list
        +
        '])'
    )

def list_tuple_bool_base_copy(input_list):
    return (
        'print('
        +
        input_list
        +
        ')'
    )

def list_tuple_bool_multiplier_random(input_list, n):
    n -= 1
    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print([tuple(bool(random.getrandbits(1)) for _ in range(len(' + input_list +'[0]))) for x in ['
        + 
        input_list 
        + 
        'for _ in range({})] for y in x] + '.format(n_1)
        + 
        input_list
        +
        ' + [tuple(bool(random.getrandbits(1)) for _ in range(len(' + input_list +'[0])))  for x in ['
        +
        input_list
        +
        'for _ in range({})] for y in x])'.format(n_2)
    )
