# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import numpy as np
import re

def equality_complexities(complexity1: str, complexity2: str):
    """
    Checks if two complexities are equal.
    This function first harmonizes the input complexities and then checks if they are equal.
    It also checks if the complexities are equivalent by checking if they are in a list of known equivalent pairs.
    Args:
        complexity1 (str): The first complexity to compare.
        complexity2 (str): The second complexity to compare.
    Returns:
        bool: True if the complexities are equal or equivalent, False otherwise.
    Notes:
        The function uses a list of known equivalent pairs to check for equivalence.
        This list can be extended to include more equivalent pairs.
    Examples:
        >>> equality_complexities('O(n)', 'O(n)')
        True
        >>> equality_complexities('O(n^2)', 'O(n)')
        False
        >>> equality_complexities('var_1*logvar_1*var_2', 'var_1*var_2*logvar_2')
        True
    """

    complexity1 = harmonize_complexity(complexity1)
    complexity2 = harmonize_complexity(complexity2)

    equivalent_pair_list = [
        ('var_1*logvar_1*var_2', 'var_1*var_2*logvar_2'),
        ('var_1*logvar_1+var_2', 'var_1+var_2*logvar_2'),
        ('var_1*logvar_2', 'logvar_1*var_2'),

        ('var_1*logvar_1+var_2+var_3', 'var_1+var_2*logvar_2+var_3'),
        ('var_1*logvar_1+var_2+var_3', 'var_1+var_2+var_3*logvar_3'),
        ('var_1+var_2*logvar_2+var_3', 'var_1+var_2+var_3*logvar_3'),

        ('var_1*logvar_1+var_2*logvar_2+var_3', 'var_1+var_2*logvar_2+var_3*logvar_3'),
        ('var_1*logvar_1+var_2*logvar_2+var_3', 'var_1*logvar_1+var_2+var_3*logvar_3'),
        ('var_1+var_2*logvar_2+var_3*logvar_3', 'var_1*logvar_1+var_2+var_3*logvar_3'),

        ('var_1*var_2+var_3','var_1+var_2*var_3'),
        ('var_1*logvar_1*var_2+var_3^2','var_1^2+var_2*logvar_2*var_3'),
        ('var_1*logvar_2*logvar_3*var_4','var_1*var_2*logvar_3*logvar_4'),



        ('var_1^2+var_2', 'var_1+var_2^2'),
        ('var_1^2*var_2', 'var_1*var_2^2'),
        ('var_1^2+logvar_2', 'logvar_1+var_2^2'),
        ('var_1^2*logvar_2', 'logvar_1*var_2^2'),

        ('var_1*logvar_2*logvar_3*logvar_4','logvar_1*logvar_2*logvar_3*var_4'),
        ('var_1*logvar_1+var_2^2','var_1^2+var_2*logvar_2'),

        ('logvar_1*logvar_2*logvar_3*var_4','logvar_1*var_2*logvar_3*logvar_4'),

        ('var_1*logvar_2*var_3*logvar_4','logvar_1*logvar_2*var_3*var_4'),

        ('logvar_1*logvar_2*var_3','var_1*logvar_2*logvar_3'),

        ('var_1+var_2+var_3*logvar_3+var_4','var_1+var_2+var_3+var_4*logvar_4'),
        ('var_1+var_2*logvar_2^2','var_1*logvar_1^2+var_2'),
        ('logvar_1*logvar_2*var_3','logvar_1*var_2*logvar_3'),
        ('logvar_1*var_2^2*logvar_3','logvar_1*var_2*logvar_3^2'),

        ('logvar_1*var_2*logvar_3','var_1*logvar_2*logvar_3'),
        ('var_1+var_2*logvar_3','logvar_1*var_2+var_3'),
        ('var_1*logvar_1^2+var_2+var_3','var_1+var_2*logvar_2^2+var_3'),
        ('logvar_1+var_2+var_3','var_1+var_2+logvar_3'),
        ('logvar_1*logvar_2*var_3*logvar_4','logvar_1*var_2*logvar_3*logvar_4'),
        ('var_1*logvar_2^2','logvar_1*var_2^2'),
        ('logvar_1*logvar_2*logvar_3*var_4','logvar_1*logvar_2*var_3*logvar_4'),
        ('var_1+var_2*logvar_1+var_2+var_3','var_1+var_2+var_3*logvar_2+var_3'),
        ('var_1+var_2^2+var_3','var_1^2+var_2+var_3'),
        ('var_1^2+var_2+var_3^2','var_1+var_2^2+var_3^2'),
        ('logvar_1*logvar_2*var_3^2','logvar_1*var_2*logvar_3^2'),
        ('var_1^2+var_2*logvar_2*var_3','var_1^2+var_2*var_3*logvar_3'),
        ('var_1*var_2*logvar_3*logvar_4','logvar_1*logvar_2*var_3*var_4'),
        ('var_1*var_2*logvar_2+var_3','var_1*var_2+var_3*logvar_3'),
        ('var_1^2+var_2*var_3','var_1*var_2+var_3^2'),
        ('var_1*var_2*var_3^2','var_1*var_2^2*var_3'),
        ('var_1*var_2^2*var_3','var_1^2*var_2*var_3'),
        ('var_1+var_2^2+var_3','var_1+var_2+var_3^2'),
        ('var_1^2*logvar_2*var_3','var_1^2*var_2*logvar_3'),
        ('var_1*var_2+var_3*logvar_2+var_3','var_1+var_2*logvar_1+var_2*var_3'),
        ('logvar_1*var_2*var_3','var_1*logvar_2*var_3'),
        ('var_1*var_2*var_3^2','var_1^2*var_2*var_3'),
        ('var_1*logvar_1*var_2*var_3','var_1*var_2*var_3*logvar_3'),
        ('var_1*logvar_1+var_2^2','var_1+var_2*logvar_2^2'),
        ('var_1^2*var_2*logvar_2','var_1*logvar_1*var_2^2'),
        ('var_1*var_2+var_3*var_4','var_1+var_2*var_3*var_4'),

    ]           

    if ((complexity1, complexity2) in equivalent_pair_list) or ((complexity2, complexity1) in equivalent_pair_list):
        return True
        
    return complexity1 == complexity2

def harmonize_complexity(complexity: str):
    """
    Harmonizes a complexity string by removing unnecessary characters, replacing equivalent symbols, and normalizing variable names.
    Args:
        complexity (str): The complexity string to harmonize.
    Returns:
        str: The harmonized complexity string.
    Notes:
        This function performs several steps to harmonize the complexity string:
            1. Replaces '**' with '^' and other equivalent symbols.
            2. Removes unnecessary characters such as parentheses and underscores.
            3. Normalizes variable names by replacing them with 'var_<number>'.
            4. Expands expressions by inserting '*' between variables and symbols.
    Examples:
        >>> harmonize_complexity('O(n^2)')
        'var_1^2'
        >>> harmonize_complexity('O(n log n)')
        'var_1*logvar_1'
        >>> harmonize_complexity('O(n*m)')
        'var_1*var_2'
    """

    symbols_to_keep = ['*', '+', 'log', '^2', '^3', '^4']
    symbols_to_delete = ['(', ')', '_']
    complexity = complexity.replace('**', '^')

    # new innovations
    complexity = complexity.replace('\\times', '*')

    complexity = complexity.replace("\u202f", "")
    complexity = complexity.replace("²", '^2')
    complexity = complexity.replace("\\", "")
    complexity = complexity.replace("×", "*")

    temp_complexity = ''

    if complexity[0:2].lower() == 'o(':
        temp_complexity = complexity[2:]
    else:
        temp_complexity = complexity

    complexity = temp_complexity
    temp_complexity = ''

    for x in complexity.lower():
        if x not in symbols_to_delete:
            temp_complexity += x

    complexity = temp_complexity
    temp_complexity = ''

    if complexity == '1':
        return '1'

    element_list = []
    previous_i = None
    i = 0
    while i < len(complexity):
        for symbol in symbols_to_keep:
            if complexity[i:i+len(symbol)] == symbol:
                if previous_i is not None:
                    if complexity[previous_i:i].isnumeric() and symbol == '*':
                        previous_i = None
                        i += len(symbol)
                        break

                    element_list.append(complexity[previous_i:i])
                    previous_i = None
                element_list.append(complexity[i:i+len(symbol)])
                i += len(symbol)
                break
        else:
            if previous_i is None:
                previous_i = i
            i += 1

    if previous_i is not None:
        element_list.append(complexity[previous_i:])
        previous_i = None

    letter_list = ['n', 'm', 'p', 'q', 'k', 'l', 'u', 'v', 'w']

    element_list_expanded = []
    for i in range(len(element_list)):
        if element_list[i] in symbols_to_keep:
            # current is a symbol
            element_list_expanded.append(element_list[i])

        else:
            # current is not a symbol
            if all([letter in letter_list for letter in element_list[i]]):
                element_list_expanded += list('*'.join(list(element_list[i])))
            else:
                element_list_expanded.append(element_list[i])

            if (i < len(element_list) - 1) and (element_list[i+1] == 'log'):
                element_list_expanded += ['*']

    element_list = element_list_expanded
    element_list_expanded = []

    # now we can normalize by varible name
    counter_ = 0

    def next_variable_name():
        nonlocal counter_
        counter_ += 1
        return 'var_' + str(counter_)
    
    element_to_anonymized_variable = collections.defaultdict(next_variable_name)

    temp_complexity = ''
    for element in element_list:
        if element in symbols_to_keep:
            temp_complexity += element
        else:
            temp_complexity += element_to_anonymized_variable[element]

    complexity = temp_complexity

    return complexity

def map_true_complexity(complexity):
    """
    Maps a given complexity string to its standardized form.
    Args:
        complexity (str): The complexity string to map.
    Returns:
        str: The mapped complexity string in its standardized form.
    Notes:
        This function uses a dictionary to map known complexities to their standardized forms.
        If the input complexity is not found in the dictionary, it is returned unchanged.
    Examples:
        >>> map_true_complexity('o(m*n)')
        'o(n*m)'
        >>> map_true_complexity('o(nlogn)')
        'o(n*logn)'
        >>> map_true_complexity('o(log(n))')
        'o(logn)'
    """
    
    complexity_dict = {
        'o(m*n)': 'o(n*m)',
        'o(nm)': 'o(n*m)',
        'o(nlogn)': 'o(n*logn)',
        'o(log(n))': 'o(logn)',
        'o(mn)': 'o(n*m)',
        'o(n2)': 'o(n^2)',
        'o(n*n)': 'o(n^2)',
    }
    return complexity_dict.get(complexity, complexity)

def get_complexity_order(complexity):
    complexity = harmonize_complexity(complexity)

    if complexity == '1':
        return 0
    
    if '**' in complexity:
        raise Exception('wrong format')
    
    return max([
        sum([
            ((y.count('^2') * 2 + y.count('^3') * 3 + y.count('log') * 0.5) 
            if (y.count('^2') * 2 + y.count('^3') * 3 + y.count('log') * 0.5) > 0
            else 1)
            for y in x.split('*')
        ])
        for x in complexity.split('+')
    ])

def get_number_variables(complexity):
    complexity = harmonize_complexity(complexity)
    return max([int(x.replace('var_', '')) for x in re.findall(r'var_[1-9]', complexity)] + [0])