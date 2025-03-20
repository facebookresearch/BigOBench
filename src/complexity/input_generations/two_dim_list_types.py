# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

############################################## 2D LIST TYPES ##############################################

# def list_multiplier(input_list, n):
#     return 'print([random.randint(0, {}) for x in ['.format(n) + input_list + ' for _ in range({})] for y in x])'.format(n)

def list_list_multiplier_identity(input_list_list, n, print_ = True):
    return ('print(' if print_ else '(') + '[[w for v in u for w in v] for u in [[x for _ in range({})] for x in '.format(n) + input_list_list + ']])'

def list_list_string_base_identity(input_list_list):
    return (
        'print([["aaaaaa" for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range(len('
        +
        input_list_list
        +
        '))])'
    )

def list_list_string_base_random(input_list_list):
    return (
        'print([["".join([random.choice(string.ascii_letters.lower()) for _ in range(random.randint(3, 10))]) for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range(len('
        +
        input_list_list
        +
        '))])'
    )

def list_list_string_base_copy(input_list_list):
    return (
        'print('
        +
        input_list_list
        +
        ')'
    )

def list_list_string_multiplier_random_dim2(input_list_list, n, print_ = True):
    n -= 1

    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        ('print(' if print_ else '(')
        +
        '[[w for v in u for w in v] for u in [[["".join([random.choice(string.ascii_letters.lower())'
        +
        ' for _ in range(random.randint(3, {}))]) '.format(max(n, 3))
        +
        'for _ in range(len(x))] for _ in range({})] + [x] + '.format(n_1)
        +
        '[["".join([random.choice(string.ascii_letters.lower()) for _ in range(random.randint(3, {}))]) for _ in '.format(max(n, 3))
        +
        'range(len(x))] for _ in range({})] for x in '.format(n_2)
        +
        input_list_list
        +
        ']]'
        +
        ')'
    )

def list_list_int_base_identity(input_list_list):
    return (
        'print([[9998 for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range(len('
        +
        input_list_list
        +
        '))])'
    )

def list_list_int_base_random(input_list_list):
    return (
        'print([[random.randint(0, 1000) for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range(len('
        +
        input_list_list
        +
        '))])'
    )

def list_list_int_base_copy(input_list_list):
    return (
        'print('
        +
        input_list_list
        +
        ')'
    )


def list_list_int_multiplier_random_dim2(input_list_list, n, print_ = True):
    n -= 1

    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        ('print(' if print_ else '(')
        +
        '[[w for v in u for w in v] for u in [[[random.randint(1, {})'.format(max(n, 1))
        +
        ' for _ in range(len(x))] for _ in range({})] + [x] + '.format(n_1)
        +
        '[[random.randint(1, {}) for _ in '.format(max(n, 1))
        +
        'range(len(x))] for _ in range({})] for x in '.format(n_2)
        +
        input_list_list
        +
        ']]'
        +
        ')'
    )

def list_list_float_base_identity(input_list_list):
    return (
        'print([[9998.2425434534 for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range(len('
        +
        input_list_list
        +
        '))])'
    )

def list_list_float_base_random(input_list_list):
    return (
        'print([[random.uniform(0, 1000) for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range(len('
        +
        input_list_list
        +
        '))])'
    )

def list_list_float_base_copy(input_list_list):
    return (
        'print('
        +
        input_list_list
        +
        ')'
    )

def list_list_float_multiplier_random_dim2(input_list_list, n, print_ = True):
    n -= 1

    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        ('print(' if print_ else '(')
        +
        '[[w for v in u for w in v] for u in [[[random.uniform(0, {})'.format(n)
        +
        ' for _ in range(len(x))] for _ in range({})] + [x] + '.format(n_1)
        +
        '[[random.uniform(0, {}) for _ in '.format(n)
        +
        'range(len(x))] for _ in range({})] for x in '.format(n_2)
        +
        input_list_list
        +
        ']]'
        +
        ')'
    )

def list_list_bool_base_identity(input_list_list):
    return (
        'print([[True for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range(len('
        +
        input_list_list
        +
        '))])'
    )

def list_list_bool_base_random(input_list_list):
    return (
        'print([[bool(random.getrandbits(1)) for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range(len('
        +
        input_list_list
        +
        '))])'
    )

def list_list_bool_base_copy(input_list_list):
    return (
        'print('
        +
        input_list_list
        +
        ')'
    )

def list_list_bool_multiplier_random_dim2(input_list_list, n, print_ = True):
    n -= 1

    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        ('print(' if print_ else '(')
        +
        '[[w for v in u for w in v] for u in [[[bool(random.getrandbits(1))'
        +
        ' for _ in range(len(x))] for _ in range({})] + [x] + '.format(n_1)
        +
        '[[bool(random.getrandbits(1)) for _ in '
        +
        'range(len(x))] for _ in range({})] for x in '.format(n_2)
        +
        input_list_list
        +
        ']]'
        +
        ')'
    )

def list_list_string_multiplier_random_dim1(input_list_list, n):
    n -= 1

    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print('
        +
        '[["".join([random.choice(string.ascii_letters.lower()) for _ in range(random.randint(3, {}))]) for _ in range(len('.format(max(n, 3))
        +
        input_list_list
        +
        '[0]))] for _ in range({}) for _ in range(len('.format(n_1)
        +
        input_list_list
        +
        '))] + '
        +
        input_list_list
        +
        '+ [["".join([random.choice(string.ascii_letters.lower()) for _ in range(random.randint(3, {}))]) for _ in range(len('.format(max(n, 3))
        +
        input_list_list
        +
        '[0]))] for _ in range({}) for _ in range(len('.format(n_2)
        +
        input_list_list
        +
        '))]'
        +
        ')'
    )

def list_list_int_multiplier_random_dim1(input_list_list, n):
    n -= 1

    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print('
        +
        '[[random.randint(1, {}) for _ in range(len('.format(max(n, 1))
        +
        input_list_list
        +
        '[0]))] for _ in range({}) for _ in range(len('.format(n_1)
        +
        input_list_list
        +
        '))] + '
        +
        input_list_list
        +
        '+ [[random.randint(1, {}) for _ in range(len('.format(max(n, 1))
        +
        input_list_list
        +
        '[0]))] for _ in range({}) for _ in range(len('.format(n_2)
        +
        input_list_list
        +
        '))]'
        +
        ')'
    )

def list_list_float_multiplier_random_dim1(input_list_list, n):
    n -= 1

    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print('
        +
        '[[random.uniform(0, {}) for _ in range(len('.format(n)
        +
        input_list_list
        +
        '[0]))] for _ in range({}) for _ in range(len('.format(n_1)
        +
        input_list_list
        +
        '))] + '
        +
        input_list_list
        +
        '+ [[random.uniform(0, {}) for _ in range(len('.format(n)
        +
        input_list_list
        +
        '[0]))] for _ in range({}) for _ in range(len('.format(n_2)
        +
        input_list_list
        +
        '))]'
        +
        ')'
    )

def list_list_bool_multiplier_random_dim1(input_list_list, n):
    n -= 1

    n_1 = n // 2
    n_2 = n - n // 2 
    assert n_1 + n_2 == n

    return (
        'print('
        +
        '[[bool(random.getrandbits(1)) for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range({}) for _ in range(len('.format(n_1)
        +
        input_list_list
        +
        '))] + '
        +
        input_list_list
        +
        '+ [[bool(random.getrandbits(1)) for _ in range(len('
        +
        input_list_list
        +
        '[0]))] for _ in range({}) for _ in range(len('.format(n_2)
        +
        input_list_list
        +
        '))]'
        +
        ')'
    )
