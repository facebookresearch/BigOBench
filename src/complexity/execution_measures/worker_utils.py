# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import ast


def get_import_statements(correct_nlogn):
    """
    Returns a string containing common import statements used in Python programming.
    If correct_nlogn is True, the function also includes an implementation of the merge sort algorithm.
    Args:
        correct_nlogn (bool): Whether to include the merge sort implementation.
    Returns:
        str: A string containing the import statements.
    """

    import_statements = (
        "import sys\n"
        "import string\n"
        "import time\n"
        "import itertools\n"
        "from itertools import accumulate, product, permutations, combinations\n"
        "import collections\n"
        "from collections import Counter, OrderedDict, deque, defaultdict, ChainMap\n"
        "import functools\n"
        "from functools import *\n"
        "from functools import lru_cache\n"
        "import math\n"
        "from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\n"
        "import fractions\n"
        "from typing import List, Tuple\n"
        "import numpy as np\n"
        "import random\n"
        "import heapq\n"
        "from math import *\n"
        "from heapq import *\n"
        "from typing import *\n"
        "import tracemalloc\n"
        "import cProfile\n"
        "random.seed(0)\n"
        "np.random.seed(0)\n"
    )

    if correct_nlogn:
        import_statements += (
            '\nimport operator\n\ndef merge41668424305621895222(left, right, compare, key):\n  '
            '  result = []\n    i, j = 0, 0\n    while i < len(left) and j < len(right):\n    '
            '    if compare(key(left[i]), key(right[j])):\n            result.append(left[i])\n  '
            '          i += 1\n        else:\n            result.append(right[j])\n    '
            '        j += 1\n    while i < len(left):\n        result.append(left[i])\n    '
            '    i += 1\n    while j < len(right):\n        result.append(right[j])\n     '
            '   j += 1\n    return result\n\ndef reverseList41668424305621895222(L):\n  '
            '  reversed_list = []\n    for i in range(len(L) - 1, -1, -1):\n      '
            '  reversed_list.append(L[i])\n\n    return reversed_list\n\ndef mergeSort41668424305621895222(L,'
            ' compare=operator.lt, key=lambda x: x, reverse=False):\n    #b = sum([sum([i for i in range(1)])'
            ' for _ in range(1)])\n    #a = b\n    #a + a\n    try:\n        len(L)\n        L[0]\n    except:\n'
            '        L = list(L)\n        \n    if len(L) < 2:\n        return L[:] if not reverse else'
            ' reverseList41668424305621895222(L[:])\n    else:\n        middle = int(len(L) / 2)\n    '
            '    left = mergeSort41668424305621895222(L[:middle], compare, key)\n    '
            '    right = mergeSort41668424305621895222(L[middle:], compare, key)\n     '
            '   return merge41668424305621895222(left, right, compare, key) if not reverse else'
            ' reverseList41668424305621895222(merge41668424305621895222(left, right, compare, key))\n  '
            '  \nsorted = mergeSort41668424305621895222\n\n'
        )

    return import_statements


def preprocess_code_content(code_content):
    """
    Preprocesses Python code content by removing unnecessary lines and replacing certain keywords.
    This function removes lines containing '__main__', 'print', 'assert', and references to 'python'. 
    It also replaces 'print' and 'assert' statements with 'pass' to prevent execution. 
    Additionally, it standardizes indentation by replacing tabs with four spaces.
    Args:
        code_content (str): The Python code content to be preprocessed.
    Returns:
        str: The preprocessed Python code content.
    """

    replace_pair_list = [
        ('python []', ''),
        ('PYTHON []', ''),
        ('python\n', '\n'),
        ('Python []', ''),
    ]

    code_content = code_content.replace('`', '')

    for replace_pair in replace_pair_list:
        code_content = code_content.replace(replace_pair[0], replace_pair[1])

    code_content_list = code_content.split('\n')

    for i, code_content_element in enumerate(code_content_list):
        if '__main__' in code_content_element and '__name__' in code_content_element:
            code_content_list[i] = ''
            break

        if 'print' in code_content_element:
            code_content_list[i] = code_content_element[:code_content_element.index(
                'print')] + 'pass'

        if 'assert' in code_content_element:
            code_content_list[i] = code_content_element[:code_content_element.index(
                'assert')] + 'pass'

        if 'python' in code_content_element.lower():
            code_content_list[i] = ''

        code_content_list[i] = code_content_list[i].replace('\\t', '    ')
        code_content_list[i] = code_content_list[i].replace('\t', '    ')

    code_content_list = code_content_list[:i+1]

    return '\n'.join(code_content_list)


def match_ast_node(node, var_names):
    """
    Match an AST node to any of the variable names in the list.
    Args:
        node (ast.AST): The AST node to match.
        var_names (list[str]): A list of variable names.
    Returns:
        bool: True if the node matches any of the variable names, False otherwise.
    """
    # Base case: If the node is a Name, check if it matches any of the variable names
    if isinstance(node, ast.Name):
        return node.id in var_names

    # If the node is an Attribute, recursively check its value and attr
    elif isinstance(node, ast.Attribute):
        # Check if the attribute name matches any of the variable names
        if isinstance(node.value, ast.Name):
            return node.value.id + '.' + node.attr in var_names

        elif isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name):
                return node.value.value.id + '.' + node.value.attr + '.' + node.attr in var_names

    return False


def remove_input_assignments(node):
    """
    Remove any line that assigns something to the variable input.
    Args:
        node (ast.AST): The root node of the AST.
    Returns:
        ast.AST: The modified AST with input assignments removed.
    """
    class InputAssignmentRemover(ast.NodeTransformer):
        def visit_Assign(self, node):
            trigger_list = [
                'input',
                "sys.stdin",
                "sys.stdout",
                "stdin",
                "stdout",
                "sys.stdin.readline",
                "sys.stdout.readline",
                "stdin.readline",
                "stdout.readline"
            ]

            for target in node.targets:
                if isinstance(target, ast.Name) or isinstance(target, ast.Attribute):
                    if match_ast_node(target, trigger_list):
                        return None

                elif isinstance(target, (ast.Tuple, ast.List)):
                    new_elts = [elt for elt in target.elts if match_ast_node(
                        elt, trigger_list)]
                    if new_elts:
                        return None

            return self.generic_visit(node)

    remover = InputAssignmentRemover()
    return remover.visit(node)


def remove_stdin_readline_imports(node):
    """
    Remove any line that imports stdin or readline.
    Args:
        node (ast.AST): The root node of the AST.
    Returns:
        ast.AST: The modified AST with stdin and readline imports removed.
    """
    class StdinReadlineImportRemover(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module in ['sys', 'io', "sys.stdin"] and any(alias.name in ['stdin', 'readline'] for alias in node.names):
                return None  # Remove this import
            return self.generic_visit(node)

    remover = StdinReadlineImportRemover()
    return remover.visit(node)


def preprocess_code_content_with_input(code_content):
    """
    Preprocesses Python code content by removing input assignments and stdin readline imports.
    This function uses the ast module to parse the code content, remove unwanted nodes, 
    and then unparse the modified tree back into code content. It also separates import star statements 
    from the rest of the code.
    Args:
        code_content (str): The Python code content to be preprocessed.
    Returns:
        tuple: A tuple containing two strings:
            - The preprocessed code content without import star statements.
            - The import star statements separated from the rest of the code.
    Note:
        This function assumes that the provided code content is valid Python syntax.
    """

    tree = ast.parse(code_content)
    modified_tree = remove_input_assignments(tree)
    code_content = ast.unparse(modified_tree)
    tree = ast.parse(code_content)
    modified_tree = remove_stdin_readline_imports(tree)
    code_content = ast.unparse(modified_tree)

    code_content_list = code_content.split('\n')

    import_star_list = []
    all_remaining_code_list = []

    for i, code_content_element in enumerate(code_content_list):
        if len(code_content_element.split()) == 4:
            a, b, c, d = code_content_element.split()
            if a == 'from' and c == "import" and d == "*":
                import_star_list.append(code_content_element)
            else:
                all_remaining_code_list.append(code_content_element)
        else:
            all_remaining_code_list.append(code_content_element)

    return '\n'.join(all_remaining_code_list), '\n'.join(import_star_list)


def wrap_code_to_get_executable_function(
        code_content, 
        arguments_to_match,
        input_handler,
    ):
    """
    This function takes in a string of Python code, a list of argument names to match, 
    and an input handler. It preprocesses the code content, removes any lines containing 
    'import *', indents the code, and wraps it in a class called ContextWrapperForTimeSpaceComplexity.
    The function then inspects all available functions and classes within this wrapper class, 
    looking for a function whose argument names match the provided list of argument names. 
    If no exact match is found, it will look for a function with the same number of arguments.
    Args:
        code_content (str): A string of Python code.
        arguments_to_match (list): A list of argument names to match.
        input_handler: An input handler (not used in this function).
    Returns:
        str: A string representing the wrapped code, including the matched function, 
             variable name to argument name mapping, and class to self-call mapping.
    Prints:
        The matched function, variable name to argument name mapping, and class to self-call mapping.
    """

    code_content = code_content.replace('\\n', '\n').replace('\\\'', '\'')
    code_content = preprocess_code_content(code_content)

    code_content = '\n'.join(
        filter(lambda x: 'import *' not in x, code_content.split('\n')))
    code_content = code_content.replace('\n', '\n    ')

    return (
        '\nclass ContextWrapperForTimeSpaceComplexity:\n    '
        +
        code_content
        +
        '\nfrom optparse import OptionParser\nimport inspect\n\n# We inspect all available functions and classes\n\nvariable_name_list = '
        +
        str(arguments_to_match)
        +
        '\nvariable_name_set = set(map(lambda x: x.lower(), variable_name_list))\nassert len(variable_name_list) == len(variable_name_set)\n\nfunction_to_execute = None\nvariable_name_to_argument_name = None\nclass_to_self_call = None\n\nclass_list = [\n    ContextWrapperForTimeSpaceComplexity, \n] + list(filter(\n    lambda x: str(x) != "<class \'type\'>", \n    map(\n        lambda x: x[1], \n        inspect.getmembers(\n            ContextWrapperForTimeSpaceComplexity, predicate=inspect.isclass\n        )\n    )\n))[::-1]\n\nfor class_ in class_list:\n    for function_name, function_ in inspect.getmembers(\n            class_, predicate=inspect.isfunction\n        )[::-1]:\n        argument_name_list = list(dict(inspect.signature(function_).parameters).keys())\n        is_self_present = \'self\' in argument_name_list\n        argument_name_list = list(filter(lambda x: x != \'self\', argument_name_list))\n        argument_name_set = set(\n            map(lambda x: x.lower(), argument_name_list)\n        )\n        if len(argument_name_list) != len(argument_name_set):\n            continue\n\n        if argument_name_set == variable_name_set:\n            variable_name_to_argument_name = dict()\n\n            for argument_name in argument_name_list:\n                for variable_name in variable_name_list:\n                    if variable_name.lower() == argument_name.lower():\n                        variable_name_to_argument_name[variable_name] = argument_name\n\n            if (\n                set(variable_name_to_argument_name.keys()) != set(variable_name_list)\n            ) or (\n                set(variable_name_to_argument_name.values()) != set(\n                    argument_name_list\n                )\n            ):\n                continue\n\n            class_to_self_call = class_ if is_self_present else None\n            function_to_execute = function_\n            break\n\n    if function_to_execute is not None:\n        break\n\n#\xa0if that fails, let\'s just try to find something that matches in number of arguments\nif function_to_execute is None:\n    for class_ in class_list:\n        for function_name, function_ in inspect.getmembers(\n                class_, predicate=inspect.isfunction\n            )[::-1]:\n            argument_name_list = list(dict(inspect.signature(function_).parameters).keys())\n            is_self_present = \'self\' in argument_name_list\n            argument_name_list = list(filter(lambda x: x != \'self\', argument_name_list))\n            argument_name_set = set(\n                map(lambda x: x.lower(), argument_name_list)\n            )\n            if len(argument_name_list) != len(argument_name_set):\n                continue\n\n            if len(argument_name_set) == len(variable_name_set):\n                variable_name_to_argument_name = dict()\n\n                for argument_name, variable_name in zip(argument_name_list, variable_name_list):\n                    variable_name_to_argument_name[variable_name] = argument_name\n\n                if (\n                    set(variable_name_to_argument_name.keys()) != set(variable_name_list)\n                ) or (\n                    set(variable_name_to_argument_name.values()) != set(\n                        argument_name_list\n                    )\n                ):\n                    continue\n\n                class_to_self_call = class_ if is_self_present else None\n                function_to_execute = function_\n                break\n\n        if function_to_execute is not None:\n            break\n\nprint(function_to_execute)\nprint(variable_name_to_argument_name)\nprint(class_to_self_call)'
    )


def add_context_to_root_functions(
    code_to_execute,
    functions_to_change
):
    """
    This function adds a context wrapper to specified root functions in a given code.
    It checks if each function to change is present in the code and then inserts the 
    context wrapper 'ContextWrapperForTimeSpaceComplexity.' before each occurrence of 
    these functions. The insertion only happens if the function is not part of a 
    definition (i.e., it's not preceded by 'def') and if it's not part of another word.
    Args:
        code_to_execute (str): The code where the context wrapper will be added.
        functions_to_change (str): A string containing the names of the functions to 
            change, separated by newline characters.
    Returns:
        str: The modified code with the context wrapper added to the specified functions.
    Raises:
        Exception: If a function to change is not found in the code.
    """

    if functions_to_change != '':
        functions_to_change = functions_to_change.split('\n')[:-1]

        for function_ in functions_to_change:
            try:
                assert function_ in code_to_execute
            except:
                raise Exception(function_ + '##########' +
                                str(functions_to_change) + '##########' + code_to_execute)

            code_to_insert = 'ContextWrapperForTimeSpaceComplexity.'

            i = 0
            code_to_execute_length = len(code_to_execute)
            assert len(function_) >= 1
            while i <= code_to_execute_length - len(function_) - 1:

                # look ahead
                if code_to_execute[i:i+len(function_)+1] == function_ + '(':

                    # look behind
                    j = i - 1
                    while j > 0 and code_to_execute[j] == ' ':
                        j -= 1

                    if (i == 0) or (
                        (code_to_execute[i-1] in [
                            ' ', '\n'
                        ] + [
                            '~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/'
                        ]) and (code_to_execute[max(0, j - 2):j+1] != 'def')
                    ):
                        code_to_execute = code_to_execute[:i] + \
                            code_to_insert + code_to_execute[i:]
                        i += len(code_to_insert)
                        code_to_execute_length += len(code_to_insert)

                i += 1

    return code_to_execute


def replace_sorting_algorithm(code_to_execute):
    """
    This function replaces all occurrences of the built-in Python sorting algorithm 
    (list.sort()) in a given code with a custom merge sort algorithm.
    The replacement only happens if the 'sort' method is called on a list, and it 
    assumes that the list is assigned to a variable. The function name of the custom 
    merge sort algorithm is 'mergeSort41668424305621895222'.
    Args:
        code_to_execute (str): The code where the sorting algorithm will be replaced.
    Returns:
        str: The modified code with the custom merge sort algorithm.
    """

    if 'sort' not in code_to_execute:
        return code_to_execute

    i = 1
    code_to_execute_length = len(code_to_execute)
    function_ = '.sort('
    while i <= code_to_execute_length - len(function_):

        # look ahead
        if code_to_execute[i:i+len(function_)] == function_:

            # look behind
            j = i - 1
            while j > 0 and (code_to_execute[j-1].isalnum() or code_to_execute[j-1] == '_'):
                j -= 1

            code_to_execute = (
                code_to_execute[:j]
                +
                code_to_execute[j:i]
                +
                ' = '
                +
                'mergeSort41668424305621895222('
                +
                code_to_execute[j:i]
                +
                ','
                +
                code_to_execute[i+len(function_):]
            )

            i += len(' = ') + len('mergeSort41668424305621895222(') + \
                len(',') + len(code_to_execute[j:i]) - len(function_)
            code_to_execute_length += len(' = ') + len('mergeSort41668424305621895222(') + len(
                ',') + len(code_to_execute[j:i]) - len(function_)

        i += 1

    return code_to_execute


def make_root_functions_accessible_inside_classes(
    code_content
):
    """
    This function takes in a string of Python code, preprocesses it, and wraps it 
    inside a class called ContextWrapperForTimeSpaceComplexity. The wrapped code 
    makes all root functions accessible as methods of this class.
    The preprocessing steps include replacing escaped newline characters with actual 
    newlines, indenting the code, replacing escaped single quotes with actual single 
    quotes, and removing any lines containing 'import *'.
    Args:
        code_content (str): A string of Python code.
    Returns:
        str: The preprocessed code wrapped inside the ContextWrapperForTimeSpaceComplexity class.
    """
        
    code_content_clean = code_content.replace(
        '\\n', '\n').replace('\n', '\n    ').replace('\\\'', '\'')
    code_content_clean = preprocess_code_content(code_content_clean)

    code_content_clean = '\n'.join(
        filter(lambda x: 'import *' not in x, code_content_clean.split('\n')))

    return (
        '\nclass ContextWrapperForTimeSpaceComplexity:\n    '
        +
        code_content_clean
        +
        '\nfrom optparse import OptionParser\nimport inspect\n\nfor function_name, function_ in inspect.getmembers(\n    ContextWrapperForTimeSpaceComplexity, predicate=inspect.isfunction\n    ):\n    print(function_name)'
    )


def make_root_objects_accessible_inside_classes(
    code_content
):
    """
    This function takes in a string of Python code, preprocesses it, and wraps it 
    inside a class called ContextWrapperForTimeSpaceComplexity. The wrapped code 
    makes all root objects (i.e., non-routine, non-class objects defined at the 
    top level of the code) accessible as attributes of this class.
    The preprocessing steps include replacing escaped newline characters with actual 
    newlines, indenting the code, replacing escaped single quotes with actual single 
    quotes, and removing any lines containing 'import *'.
    Args:
        code_content (str): A string of Python code.
    Returns:
        str: The preprocessed code wrapped inside the ContextWrapperForTimeSpaceComplexity class.
    """

    code_content_clean = code_content.replace(
        '\\n', '\n').replace('\n', '\n    ').replace('\\\'', '\'')
    code_content_clean = preprocess_code_content(code_content_clean)

    code_content_clean = '\n'.join(
        filter(lambda x: 'import *' not in x, code_content_clean.split('\n')))

    return (
        '\nclass ContextWrapperForTimeSpaceComplexity:\n    '
        +
        code_content_clean
        +
        '\nfrom optparse import OptionParser\nimport inspect\n\nfor object_name, object_ in inspect.getmembers(\n    ContextWrapperForTimeSpaceComplexity, predicate=lambda a:(not(inspect.isroutine(a))) and (not(inspect.isclass(a))) and ((not hasattr(a, "__module__")) or (a.__module__ == "__main__"))\n    ):\n    if not(object_name.startswith("__") and object_name.endswith("__")):\n        print(object_name)'
    )


def add_context_to_root_objects(
    code_to_execute,
    objects_to_change
):
    """
    This function adds a context wrapper to specified root objects in a given code.
    It checks if each object to change is present in the code and then inserts the 
    context wrapper 'ContextWrapperForTimeSpaceComplexity.' before each occurrence of 
    these objects. The insertion only happens if the object is at the start of a line 
    and not part of a definition (i.e., it's not preceded by 'def').
    Args:
        code_to_execute (str): The code where the context wrapper will be added.
        objects_to_change (str): A string containing the names of the objects to 
            change, separated by newline characters.
    Returns:
        str: The modified code with the context wrapper added to the specified objects.
    Raises:
        Exception: If an object to change is not found in the code.
    Notes:
        - The function assumes that the objects to change are exact matches, i.e., 
          they do not contain any special characters or whitespace.
        - The function does not handle cases where the objects to change are nested 
          inside other objects or functions.
    """

    code_to_execute = code_to_execute.replace('\\n', '\n')
    if objects_to_change != '':
        objects_to_change = objects_to_change.split('\n')[:-1]

        for object_ in objects_to_change:
            try:
                assert object_ in code_to_execute
            except:
                raise Exception(
                    object_ + '##########' + str(objects_to_change) + '##########' + code_to_execute)

            code_to_insert = 'ContextWrapperForTimeSpaceComplexity.'

            i = 0
            code_to_execute_length = len(code_to_execute)
            assert len(object_) >= 1
            while i <= code_to_execute_length - len(object_):

                # look ahead
                if code_to_execute[i:i+len(object_)] == object_:

                    # look behind
                    j = i - 1
                    while j > 0 and code_to_execute[j] != '\n':
                        j -= 1

                    # cases where we do not change anything
                    if (i == 0) or (j < 0) or (code_to_execute[j] != '\n') or (code_to_execute[j:j+2] != '\n ' and code_to_execute[j:j+5] != '\ndef '):
                        pass

                    else:
                        code_to_execute = code_to_execute[:i] + \
                            code_to_insert + code_to_execute[i:]
                        i += len(code_to_insert)
                        code_to_execute_length += len(code_to_insert)

                i += 1

    return code_to_execute


def execute_code_with_inputs(
    code_content,
    function_to_execute,
    variable_name_to_argument_name,
    class_to_self_call,
    variable_name_to_input_dict,
    embed_cprofile='no'
):
    """
    This function executes a given code with specified inputs and measures its time 
    and space complexity. The code is wrapped in a class called ContextWrapperForTimeSpaceComplexity.
    Args:
        code_content (str): The code to be executed.
        function_to_execute (str): The name of the function to be executed.
        variable_name_to_argument_name (dict): A dictionary mapping variable names 
            to argument names.
        class_to_self_call (str): The name of the class to be used for self-calls.
        variable_name_to_input_dict (dict): A dictionary mapping variable names to 
            input values.
        embed_cprofile (str, optional): Whether to use cProfile to measure execution 
            time. Defaults to 'no'. Can be 'within' or 'around'.
    Returns:
        tuple: A tuple containing three strings: 
            1. The wrapped code with the context wrapper.
            2. The stringified input dictionary.
            3. The closing statement that executes the function and measures time 
                and space complexity.
    Raises:
        Exception: If the execution time cannot be measured.
    Notes:
        - The function assumes that the code is valid Python code.
        - The function uses tracemalloc to measure memory usage.
        - The function uses cProfile to measure execution time if embed_cprofile is 
          'within' or 'around'.
    """
        
    assert embed_cprofile in ['no', 'within', 'around']

    code_content = code_content.replace('\\n', '\n').replace(
        '\n', '\n    ').replace('\\\'', '\'')
    code_content = preprocess_code_content(code_content)

    code_content = '\n'.join(
        filter(lambda x: 'import *' not in x, code_content.split('\n')))

    variable_name_to_input_dict_stringified = '{'
    for x, y in variable_name_to_input_dict.items():
        variable_name_to_input_dict_stringified += '"' + x + '"' + ':' + y + ','
    variable_name_to_input_dict_stringified += '}'

    closing_statement = ''

    if embed_cprofile == 'no':
        closing_statement = '\n\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ntry:\n    if class_to_self_call is None:\n        output_values = function_to_execute(\n            **varied_input_dict\n        )\n    else:\n        output_values = function_to_execute(\n            class_to_self_call(),\n            **varied_input_dict\n        )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    del output_values\n    after_memory, _ = tracemalloc.get_traced_memory()\n    print(peak_memory - current_memory + after_memory)\n\nfinally:\n    tracemalloc.stop()\n'

    elif embed_cprofile == 'within':
        closing_statement = "\n\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ncpr = cProfile.Profile()\ncpr.enable()\n\ntry:\n    if class_to_self_call is None:\n        output_values = function_to_execute(\n            **varied_input_dict\n        )\n    else:\n        output_values = function_to_execute(\n            class_to_self_call(),\n            **varied_input_dict\n        )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    del output_values\n    after_memory, _ = tracemalloc.get_traced_memory()\n    print(peak_memory - current_memory + after_memory)\n\nfinally:\n    cpr.disable()\n    tracemalloc.stop()\n\nfunction_to_execute_name = str(function_to_execute).split('.')[-1].split(' ')[0]\nfound = False\nfor entry in cpr.getstats():\n    if type(entry.code) != str:\n        function_name = entry.code.co_name\n\n        if function_name == function_to_execute_name:\n            found = True\n            print(entry.totaltime)\n            break\n\nif not found:\n    raise Exception('Could not get the execution time')\n\n"

    elif embed_cprofile == "around":
        closing_statement = "\n\ncpr = cProfile.Profile()\ncpr.enable()\n\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ntry:\n    if class_to_self_call is None:\n        output_values = function_to_execute(\n            **varied_input_dict\n        )\n    else:\n        output_values = function_to_execute(\n            class_to_self_call(),\n            **varied_input_dict\n        )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    del output_values\n    after_memory, _ = tracemalloc.get_traced_memory()\n    print(peak_memory - current_memory + after_memory)\n\nfinally:\n    tracemalloc.stop()\n    cpr.disable()\n\nfunction_to_execute_name = str(function_to_execute).split('.')[-1].split(' ')[0]\nfound = False\nfor entry in cpr.getstats():\n    if type(entry.code) != str:\n        function_name = entry.code.co_name\n\n        if function_name == function_to_execute_name:\n            found = True\n            print(entry.totaltime)\n            break\n\nif not found:\n    raise Exception('Could not get the execution time')\n\n"

    return ((
        'class ContextWrapperForTimeSpaceComplexity:\n    '
        +
        code_content
        +
        '\nfunction_to_execute = '
        +
        function_to_execute
        +
        '\nvariable_name_to_argument_name = '
        +
        variable_name_to_argument_name
        +
        '\nclass_to_self_call = '
        +
        class_to_self_call
        +
        '\nvariable_name_to_input_dict = '
    ),
        variable_name_to_input_dict_stringified,
        (
        '\nvaried_input_dict = {}\n\nfor temp_variable_name, temp_input in variable_name_to_input_dict.items():\n    varied_input_dict[variable_name_to_argument_name[temp_variable_name]] = temp_input'
        +
        closing_statement
    ))


def get_variable_name_to_input_dict_stringified(
    variable_name_to_input_dict,
):
    """
    Converts a dictionary mapping variable names to input values into a stringified format.
    Args:
        variable_name_to_input_dict (dict): A dictionary where keys are variable names and values are input values. The variable names should match the argument names of the function being executed, as determined by the `wrap_code_to_get_executable_function` function.
    Returns:
        str: A string representation of the input dictionary in the format '{"variable_name": input_value, ...}'. This stringified dictionary is then used in the `execute_code_with_inputs` function to pass the input values to the executable function.
    Notes:
        This function assumes that the input dictionary values are already stringified. It also assumes that the variable names in the dictionary match the argument names of the function being executed, as determined by the `wrap_code_to_get_executable_function` function.
    Example:
        Suppose we have a code snippet that defines a function `add(a, b)`, and we want to execute this function with inputs `a=2` and `b=3`. We would first create a dictionary `variable_name_to_input_dict = {'a': '2', 'b': '3'}`, and then pass this dictionary to this function to get the stringified dictionary `'{"a": 2, "b": 3}'`. This stringified dictionary can then be used in the `execute_code_with_inputs` function to execute the `add` function with the specified inputs.
    """
    variable_name_to_input_dict_stringified = '{'
    for x, y in variable_name_to_input_dict.items():
        variable_name_to_input_dict_stringified += '"' + x + '"' + ':' + y + ','
    variable_name_to_input_dict_stringified += '}'

    return variable_name_to_input_dict_stringified


def get_source_tuple(
    code_content,
    function_to_execute,
    variable_name_to_argument_name,
    class_to_self_call,
    embed_cprofile='no'
):
    """
    This function generates a tuple of two strings representing the source code 
    for calculating time and space complexity of a given function.
    Args:
        code_content (str): The content of the code to be executed.
        function_to_execute (callable): The function to be executed.
        variable_name_to_argument_name (dict): A dictionary mapping variable names to argument names.
        class_to_self_call (callable, optional): The class to be used for self call. Defaults to None.
        embed_cprofile (str, optional): The type of cProfile to be embedded. Defaults to 'no'.
    Returns:
        tuple: A tuple containing two strings representing the source code.
    """
    code_content = code_content.replace('\\n', '\n').replace(
        '\n', '\n    ').replace('\\\'', '\'')
    code_content = preprocess_code_content(code_content)

    code_content = '\n'.join(
        filter(lambda x: 'import *' not in x, code_content.split('\n')))

    closing_statement = ''

    if embed_cprofile == 'cprofilewithin':
        closing_statement = "\n\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ncpr = cProfile.Profile()\ncpr.enable()\n\ntry:\n    if class_to_self_call is None:\n        output_values = function_to_execute(\n            **varied_input_dict\n        )\n    else:\n        output_values = function_to_execute(\n            class_to_self_call(),\n            **varied_input_dict\n        )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    del output_values\n    after_memory, _ = tracemalloc.get_traced_memory()\n    print(peak_memory - current_memory + after_memory)\n\nfinally:\n    cpr.disable()\n    tracemalloc.stop()\n\nfunction_to_execute_name = str(function_to_execute).split('.')[-1].split(' ')[0]\nfound = False\nfor entry in cpr.getstats():\n    if type(entry.code) != str:\n        function_name = entry.code.co_name\n\n        if function_name == function_to_execute_name:\n            found = True\n            print(entry.totaltime)\n            break\n\nif not found:\n    raise Exception('Could not get the execution time')\n\n"
    elif embed_cprofile == "cprofilearound":
        closing_statement = "\n\ncpr = cProfile.Profile()\ncpr.enable()\n\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ntry:\n    if class_to_self_call is None:\n        output_values = function_to_execute(\n            **varied_input_dict\n        )\n    else:\n        output_values = function_to_execute(\n            class_to_self_call(),\n            **varied_input_dict\n        )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    del output_values\n    after_memory, _ = tracemalloc.get_traced_memory()\n    print(peak_memory - current_memory + after_memory)\n\nfinally:\n    tracemalloc.stop()\n    cpr.disable()\n\nfunction_to_execute_name = str(function_to_execute).split('.')[-1].split(' ')[0]\nfound = False\nfor entry in cpr.getstats():\n    if type(entry.code) != str:\n        function_name = entry.code.co_name\n\n        if function_name == function_to_execute_name:\n            found = True\n            print(entry.totaltime)\n            break\n\nif not found:\n    raise Exception('Could not get the execution time')\n\n"
    elif embed_cprofile == "cprofilerobust":
        closing_statement = "\nclass ClassWrapperBenchmark:\n    def benchmark_function():\n        for i in range(10000):   \n            i * i * i * i * i\n        return\n    \nfunction_to_benchmark = ClassWrapperBenchmark.benchmark_function\n# tracemalloc.start()\n# tracemalloc.clear_traces()\n# tracemalloc.reset_peak()\n\n\nbenchmark_value_list = []\nfor _ in range(25):\n    cpr = cProfile.Profile()\n    cpr.enable()\n\n    try:\n        output_values = function_to_benchmark()\n        # current_memory, peak_memory = tracemalloc.get_traced_memory()\n        del output_values\n        # after_memory, _ = tracemalloc.get_traced_memory()\n        # print(peak_memory - current_memory + after_memory)\n\n    finally:\n        cpr.disable()\n        # tracemalloc.stop()\n\n    function_to_benchmark_name = str(function_to_benchmark).split('.')[-1].split(' ')[0]\n    found = False\n    for entry in cpr.getstats():\n        if type(entry.code) != str:\n            function_name = entry.code.co_name\n\n            if function_name == function_to_benchmark_name:\n                found = True\n                benchmark_value_list.append(float(entry.totaltime))\n                break\n\n    if not found:\n        raise Exception('Could not get the execution time')\n\ntemp_runtime_benchmark_1 = np.mean(benchmark_value_list)\n\ntemp_runtime = None\ntemp_memory = None\n\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ncpr = cProfile.Profile()\ncpr.enable()\n\ntry:\n    if class_to_self_call is None:\n        output_values = function_to_execute(\n            **varied_input_dict\n        )\n    else:\n        output_values = function_to_execute(\n            class_to_self_call(),\n            **varied_input_dict\n        )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    del output_values\n    after_memory, _ = tracemalloc.get_traced_memory()\n    temp_memory = int(peak_memory - current_memory + after_memory)\n\nfinally:\n    cpr.disable()\n    tracemalloc.stop()\n\nfunction_to_execute_name = str(function_to_execute).split('.')[-1].split(' ')[0]\nfound = False\nfor entry in cpr.getstats():\n    if type(entry.code) != str:\n        function_name = entry.code.co_name\n\n        if function_name == function_to_execute_name:\n            found = True\n            temp_runtime = float(entry.totaltime)\n            break\n\nif not found:\n    raise Exception('Could not get the execution time')\n\nclass ClassWrapperBenchmark:\n    def benchmark_function():\n        for i in range(10000):   \n            i * i * i * i * i\n        return\n    \nfunction_to_benchmark = ClassWrapperBenchmark.benchmark_function\n# tracemalloc.start()\n# tracemalloc.clear_traces()\n# tracemalloc.reset_peak()\n\n\nbenchmark_value_list = []\nfor _ in range(25):\n    cpr = cProfile.Profile()\n    cpr.enable()\n\n    try:\n        output_values = function_to_benchmark()\n        # current_memory, peak_memory = tracemalloc.get_traced_memory()\n        del output_values\n        # after_memory, _ = tracemalloc.get_traced_memory()\n        # print(peak_memory - current_memory + after_memory)\n\n    finally:\n        cpr.disable()\n        # tracemalloc.stop()\n\n    function_to_benchmark_name = str(function_to_benchmark).split('.')[-1].split(' ')[0]\n    found = False\n    for entry in cpr.getstats():\n        if type(entry.code) != str:\n            function_name = entry.code.co_name\n\n            if function_name == function_to_benchmark_name:\n                found = True\n                benchmark_value_list.append(float(entry.totaltime))\n                break\n\n    if not found:\n        raise Exception('Could not get the execution time')\n\ntemp_runtime_benchmark_2 = np.mean(benchmark_value_list)\n\nprint(temp_memory)\nprint(temp_runtime/(temp_runtime_benchmark_1 * temp_runtime_benchmark_2))\n"
    else:
        closing_statement = '\n\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ntry:\n    if class_to_self_call is None:\n        output_values = function_to_execute(\n            **varied_input_dict\n        )\n    else:\n        output_values = function_to_execute(\n            class_to_self_call(),\n            **varied_input_dict\n        )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    del output_values\n    after_memory, _ = tracemalloc.get_traced_memory()\n    print(peak_memory - current_memory + after_memory)\n\nfinally:\n    tracemalloc.stop()\n'

    return ((
        'class ContextWrapperForTimeSpaceComplexity:\n    '
        +
        code_content
        +
        '\nfunction_to_execute = '
        +
        function_to_execute
        +
        '\nvariable_name_to_argument_name = '
        +
        variable_name_to_argument_name
        +
        '\nclass_to_self_call = '
        +
        class_to_self_call
        +
        '\nvariable_name_to_input_dict = '
    ),
        (
        '\nvaried_input_dict = {}\n\nfor temp_variable_name, temp_input in variable_name_to_input_dict.items():\n    varied_input_dict[variable_name_to_argument_name[temp_variable_name]] = temp_input'
        +
        closing_statement
    ))


def get_source_tuple_with_dataclass(
    code_content,
    dataclass_code,
    inputs,
    embed_cprofile='no',
    offset_input_code=False,
):
    """
    This function generates a tuple of two strings representing the source code 
    for calculating time and space complexity of a given function using a dataclass.
    Args:
        code_content (str): The content of the code to be executed.
        dataclass_code (str): The code defining the dataclass.
        inputs (str): The input string.
        embed_cprofile (str, optional): The type of cProfile to be embedded. Defaults to 'no'.
        offset_input_code (bool, optional): Whether to offset the input code. Defaults to False.
    Returns:
        tuple: A tuple containing two strings representing the source code.
    Raises:
        Exception: If the embed_cprofile is not implemented or if the dataclass cannot be made to work.
    """
    
    if offset_input_code:
        input_related_code = "\nInput.from_str(str_escaped(input_repr))\n"
        input_related_code = '\nn = int(input())\na_list = []\ntry:\n    while True:\n        a_list.append(int(input()))\n\nexcept:\n    # done reading the list\n    pass\n'

        if len(input_related_code) <= 2:
            input_related_code = '\npass\n'

    else:
        input_related_code = '\npass\n'

    code_content, import_content = preprocess_code_content_with_input(
        code_content
    )

    code_content = code_content.replace('\n', '\n    ')
    input_related_code = input_related_code.replace('\n', '\n    ')

    closing_statement = ''

    if embed_cprofile == 'cprofilewithin':
        closing_statement = "tracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ncpr = cProfile.Profile()\ncpr.enable()\n\ntry:\n    function_to_execute(\n        input\n    )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    print(peak_memory - current_memory)\n\nfinally:\n    cpr.disable()\n    tracemalloc.stop()\n\nfunction_to_execute_name = str(function_to_execute).split('.')[-1].split(' ')[(0 if '.' in str(function_to_execute) else 1)]\nfound = False\nfor entry in cpr.getstats():\n    if type(entry.code) != str:\n        function_name = entry.code.co_name\n\n        if function_name == function_to_execute_name:\n            found = True\n            print(entry.totaltime)\n            break\n\nif not found:\n    raise Exception('Could not get the execution time')"
    elif embed_cprofile == "cprofilearound":
        closing_statement = "\nassert not offset_input_code\n\ncpr = cProfile.Profile()\ncpr.enable()\n\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ntry:\n    if offset_input_code:\n        function_wrapper_for_input_related_code(\n            input\n        )\n        current_memory_input, peak_memory_input = tracemalloc.get_traced_memory()\n\n        input.reset()\n    else:\n        current_memory_input, peak_memory_input = 0, 0\n\n    function_to_execute(\n        input\n    )\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n\n    print(peak_memory - current_memory_input)\n\nfinally:\n    tracemalloc.stop()\n    cpr.disable()\n\nif offset_input_code:\n    function_to_execute_name = str(function_wrapper_for_input_related_code).split('.')[-1].split(' ')[(0 if '.' in str(function_wrapper_for_input_related_code) else 1)]\n    found = False\n    for entry in cpr.getstats():\n        if type(entry.code) != str:\n            function_name = entry.code.co_name\n\n            if function_name == function_to_execute_name:\n                found = True\n                time_1 = entry.totaltime\n                break\n                \n    if not found:\n        raise Exception('Could not get the execution time 1')\nelse:\n    time_1 = 0\n\nfunction_to_execute_name = str(function_to_execute).split('.')[-1].split(' ')[(0 if '.' in str(function_to_execute) else 1)]\nfound = False\nfor entry in cpr.getstats():\n    if type(entry.code) != str:\n        function_name = entry.code.co_name\n\n        if function_name == function_to_execute_name:\n            found = True\n            time_2 = entry.totaltime\n            break\n\nif not found:\n    raise Exception('Could not get the execution time 2')\n\nprint(time_2 - time_1)\n\n"
    elif embed_cprofile == "cprofilerobust":
        raise Exception("Not implemented")
    else:
        raise Exception('not implemented')

    return ((
        '\n'
        +
        dataclass_code
        +
        "\n"
        +
        import_content
        +
        '\ninput_str = \''
        +
        inputs
        +
        '\''
        +
        '\noffset_input_code = ' + str(offset_input_code) + '\n'
        +
        '\ndef function_wrapper_for_time_space_complexity(input):\n    '
        +
        code_content
        +
        '\ndef function_wrapper_for_input_related_code(input_repr):\n    '
        +
        input_related_code
        +
        '\nfunction_to_execute = '
        +
        'function_wrapper_for_time_space_complexity'
        +
        '\nvariable_name_to_input_dict = '
    ),
        (
        "\nclass InputIterator:\n    def __init__(self, data):\n    "
        +
        "    self.data = data.split('\\n')\n        self.data = self.data[:len(self.data) - "
        +
        "(1 if data[-len('\\n'):] == '\\n' else 0)]\n        self.index = 0\n\n    def reset(self):\n   "
        +
        "     self.index = 0\n\n    def __iter__(self):\n        return self\n    def __next__(self):\n  "
        +
        "      if self.index < len(self.data):\n            value = self.data[self.index]\n          "
        +
        "  self.index += 1\n            return value\n        else:\n            raise StopIteration\n   "
        +
        "     \n    def __call__(self):\n        return next(self)\n"
        +
        "\ninput_cls = Input.from_str(str_escaped(input_str))\n\nfound = False\nboolean_list_index ="
        +
        " 0\nboolean_list_list = [[bool((i >> j) & 1) for j in range(len(variable_name_to_input_dict))] "
        +
        "for i in range(2**len(variable_name_to_input_dict))]\n\nwhile not found and (boolean_list_index"
        +
        " < len(boolean_list_list)):\n    try:\n        boolean_list = boolean_list_list[boolean_list_index]\n "
        +
        "       for i, (key, value) in enumerate(variable_name_to_input_dict.items()):\n         "
        +
        "   setattr(input_cls, key, value if not boolean_list[i] else str(value))\n\n      "
        +
        "  input_repr = input_cls.__repr__()\n        found = True\n    except:\n        found = False\n\n "
        +
        "   boolean_list_index += 1\n\nif not found:\n    raise Exception('did not manage to make the dataclass work')\n"
        +
        "\ninput = InputIterator(input_repr)\n"
        +
        "\nimport sys\n"
        +
        "\nfrom sys import stdin\n"
        +
        "\nsys.stdin.readline = input\n"
        +
        "\nstdin.readline = input\n"
        +
        closing_statement
    ))


def generate_expansion_details_list(
    variable_type_dimension_to_expansion_details_list,
    variable_name_type_dimension_to_base_input_dict_dict,
    variable_name_ref,
    variable_type_ref,
    dimension_ref,
):
    """
    Generates a list of expansion details based on the provided reference variable and expansion rules.
    This function iterates over the expansion details associated with the reference variable's type and dimension, 
    applies filters and tags to determine which variables to include in each expansion, 
    and constructs input and multiplier dictionaries for each expansion.
    Args:
        variable_type_dimension_to_expansion_details_list (dict): A dictionary mapping variable types and dimensions to lists of expansion details.
        variable_name_type_dimension_to_base_input_dict_dict (dict): A dictionary mapping variable names, types, and dimensions to base input dictionaries.
        variable_name_ref (str): The name of the reference variable.
        variable_type_ref (str): The type of the reference variable.
        dimension_ref (str or int): The dimension of the reference variable.
    Returns:
        list: A list of expansion details, where each detail is a dictionary containing:
            - 'variable_name_to_input_dict': A dictionary mapping variable names to input values.
            - 'variable_name_to_multiplier_dict': A dictionary mapping variable names to multipliers.
            - 'tag_list': A list of tags associated with the expansion.
            - 'id_': An identifier for the expansion.
            - 'priority': A priority value for the expansion.
    Raises:
        AssertionError: If the reference variable is not found in the variable_name_type_dimension_to_base_input_dict_dict.
    """

    assert (variable_name_ref, variable_type_ref,
            dimension_ref) in variable_name_type_dimension_to_base_input_dict_dict.keys()

    expansion_details_list = []

    for i, expansion_details in enumerate(variable_type_dimension_to_expansion_details_list[
        (variable_type_ref, dimension_ref)
    ]):
        ref_base = expansion_details['ref_base']  # it'a triplet
        ref_multiplier = expansion_details['ref_multiplier']
        target_filter_base_multiplier_list = expansion_details['target_filter_base_multiplier_list']
        tag = expansion_details['tag']
        id_ = expansion_details['id_']
        tag_min_count = expansion_details['tag_min_count']
        priority = expansion_details['priority']
        tag_list = []

        variable_name_to_input_dict = dict()
        variable_name_to_multiplier_dict = dict()

        variable_name_to_multiplier_dict[variable_name_ref] = ref_multiplier

        for (
            variable_name_targ, variable_type_targ, dimension_targ
        ), base_input_dict in variable_name_type_dimension_to_base_input_dict_dict.items():

            if tag(variable_name_ref, variable_type_ref, dimension_ref, variable_name_targ, variable_type_targ, dimension_targ):
                tag_list.append(str_variable_name_type_dimension(
                    variable_name_targ, variable_type_targ, dimension_targ))

            if (variable_name_ref, variable_type_ref, dimension_ref) == (variable_name_targ, variable_type_targ, dimension_targ):
                variable_name_to_input_dict[variable_name_ref] = base_input_dict[ref_base[:2]]

        if len(tag_list) < tag_min_count:
            continue

        for target_filter_base_multiplier in target_filter_base_multiplier_list:
            target_filter = target_filter_base_multiplier['target_filter']
            target_base = target_filter_base_multiplier['target_base']
            target_multiplier = target_filter_base_multiplier['target_multiplier']

            for (
                variable_name_targ, variable_type_targ, dimension_targ
            ), base_input_dict in variable_name_type_dimension_to_base_input_dict_dict.items():
                if target_filter(
                    variable_name_ref, variable_type_ref, dimension_ref, variable_name_targ, variable_type_targ, dimension_targ
                ) and (variable_name_ref != variable_name_targ):
                    variable_name_to_input_dict[variable_name_targ] = base_input_dict[target_base[:2]]
                    variable_name_to_multiplier_dict[variable_name_targ] = target_multiplier

        expansion_details_list.append(
            {
                'variable_name_to_input_dict': variable_name_to_input_dict,
                'variable_name_to_multiplier_dict': variable_name_to_multiplier_dict,
                'tag_list': tag_list,
                'id_': id_,
                'priority': priority,
            }
        )

    return expansion_details_list


def str_variable_name_type_dimension(variable_name, variable_type, dimension):
    """
    Returns a string representation of a variable's name, type, and dimension.
    The returned string is a concatenation of the variable's name, type, and dimension, 
    separated by '####'.
    Args:
        variable_name (str): The name of the variable.
        variable_type (str): The type of the variable.
        dimension (str or int): The dimension of the variable.
    Returns:
        str: A string representation of the variable's name, type, and dimension.
    """

    return '####'.join([
        str(variable_name),
        str(variable_type),
        str(dimension),
    ])
