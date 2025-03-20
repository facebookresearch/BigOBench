# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import reduce

class ComplexityCandidate:
    """
    Stores all the information of a complexity candidate.
    A complexity candidate comes from fitted info on a value run (runtime values or memory values).
    Attributes:
        class_ (str): The class of the complexity candidate (e.g. 'o(1)', 'O(n)', etc.). Defaults to 'o(1)'.
        order (int): The order of the complexity candidate. Defaults to 0.
        class_backup (str): The backup class of the complexity candidate. Defaults to 'o(1)'.
        order_backup (int): The backup order of the complexity candidate. Defaults to 0.
        max_time (float): The maximum time for the complexity candidate. Defaults to 0.
        priority (int): The priority of the complexity candidate. Defaults to 1.
        found_peak (bool): Whether a peak was found in the data. Defaults to False.
        coeff (float): The coefficient of the complexity candidate. Defaults to 1.
        coeff_backup (float): The backup coefficient of the complexity candidate. Defaults to 1.
    Methods:
        __init__: Initializes a new instance of the ComplexityCandidate class.
    Examples:
        >>> candidate = ComplexityCandidate(class_='O(n)', order=1, max_time=10)
        >>> print(candidate.class_)
        O(n)
        >>> print(candidate.order)
        1
        >>> print(candidate.max_time)
        10
    """

    def __init__(
        self, 
        class_ = 'o(1)', 
        order = 0,
        class_backup = 'o(1)',
        order_backup = 0,
        max_time = 0,
        priority = 1,
        found_peak = False,
        coeff = 1,
        coeff_backup = 1,
    ):
        self.class_ = class_
        self.order = order
        self.class_backup = class_backup
        self.order_backup = order_backup
        self.max_time = max_time
        self.priority = priority
        self.found_peak = found_peak
        self.coeff = coeff
        self.coeff_backup = coeff_backup



class NodeComplexity:
    """
    Represents a node in the complexity graph.
    Attributes:
        variable_name_type_dimension (str): The name of the variable associated with this node.
        complexity_candidate_list (list): A list of complexity candidates for this node.
        complexity_candidate (ComplexityCandidate): The selected complexity candidate for this node.
        sum_order (int): The sum of the orders of the complexity candidates.
        max_time (float): The maximum time for this node.
        is_a_node (bool): Whether this object is a node or not.
        print_info (bool): Whether to print information during calculations.
    Methods:
        __init__: Initializes a new instance of the NodeComplexity class.
        add_complexity: Adds a complexity candidate to the list.
        self_set_complexity: Sets the complexity candidate for this node based on the given parameters.
        remove_negligeable_complexity: Removes negligible complexity from this node.
        remove_constant_complexity: Removes constant complexity from this node.
        self_adjust_constant_complexity: Adjusts the constant complexity for this node.
        self_adjust_group_operations: Adjusts the group operations for this node.
        get_encapsulated_variable_name_type_dimension: Returns the encapsulated variable name and type dimension.
        format_complexity: Formats the complexity of this node as a string.
    Examples:
        >>> node = NodeComplexity('variable_name')
        >>> node.add_complexity(ComplexityCandidate(class_='o(n)', order=1))
        >>> node.self_set_complexity(max_time_rate=0.8, elect_function=np.argmin)
        >>> print(node.format_complexity(letter_list=['n', 'm', 'k'], next_letter_index=0))
        ('n', 1, 1.0, 1.0)
    """

    def __init__(self, variable_name_type_dimension):
        self.variable_name_type_dimension = variable_name_type_dimension

        self.complexity_candidate_list = []
        self.complexity_candidate = None
        self.sum_order = None
        self.max_time = None
        self.is_a_node = True

        self.print_info = False

    def add_complexity(self, complexity_candidate):
        self.complexity_candidate_list.append(complexity_candidate)

    def self_set_complexity(
        self,
        max_time = None, 
        fix_negligeable_complexity = None,
        max_time_rate = 0.8, 
        elect_function = np.argmin, 
        fix_constant_complexity = False,
    ):
        self.complexity_candidate = compute_complexity_candidate(
            complexity_candidate_list = self.complexity_candidate_list, 
            max_time_rate = max_time_rate, 
            elect_function = elect_function, 
            use_backup = fix_constant_complexity,
        )
        
        if self.print_info:
            pass

        self.sum_order = self.complexity_candidate.order
        self.max_time = self.complexity_candidate.max_time

    def remove_negligeable_complexity(self, max_time):
        if self.print_info:
            print('remove negligeable complexity', self.variable_name_type_dimension)
        if ((self.max_time / max_time) < 0.2) and (self.complexity_candidate.class_ != 'o(1)'):
            self.complexity_candidate.class_ = 'o(1)'
            self.complexity_candidate.coeff = self.max_time
            self.complexity_candidate.order = 0
            self.sum_order = 0

    def remove_constant_complexity(self, max_time_rate, elect_function):
        self.self_set_complexity(
            max_time = None,
            fix_negligeable_complexity = None,
            max_time_rate = max_time_rate, 
            elect_function = elect_function, 
            fix_constant_complexity=True
        )

    def self_adjust_constant_complexity(
        self, 
        max_time = None, 
        fix_negligeable_complexity = True,
        max_time_rate = 0.8, 
        elect_function = np.argmin, 
        fix_constant_complexity = True,
    ):
        pass
        
    def self_adjust_group_operations(
        self, 
        max_time = None, 
        fix_negligeable_complexity = True,
        max_time_rate = 0.8, 
        elect_function = np.argmin, 
        fix_constant_complexity = True,
    ):
        pass

    def get_encapsulated_variable_name_type_dimension(self):
        return [self.variable_name_type_dimension]

    def format_complexity(
        self, 
        letter_list = ['n', 'm', 'k', 'l', 'u', 'v', 'w'], 
        next_letter_index = 0,
    ):
        mapping_complexity = {
            'o(n)': '{}',
            'o(n^2)': '{}**2',
            'o(logn)': 'log{}',
            'o(nlogn)': '{}log{}',
            'o(n^3)': '{}**3',
            'o(1)': '1',
        }

        return_values =  mapping_complexity[
            self.complexity_candidate.class_
        ].format(letter_list[next_letter_index], letter_list[next_letter_index], letter_list[next_letter_index]), next_letter_index + (
            0 
            if (self.complexity_candidate.class_) == 'o(1)'
            else 1 
        ), self.complexity_candidate.coeff, self.complexity_candidate.coeff

        if self.print_info:
            print('complexity:', self.get_encapsulated_variable_name_type_dimension(), return_values[0])

        return return_values

class GroupComplexity:
    """
    Represents a group of complexities.
    Attributes:
        group_or_node_list (list): A list of groups or nodes in this group.
        operation (str): The operation to apply to the complexities in this group.
        coeff_operation (function): The coefficient operation to apply to the complexities in this group.
        encapsulating_coeff_operation (function): The encapsulating coefficient operation to apply to the complexities in this group.
        encapsulating_template (str): The template to use for encapsulating the complexities in this group.
        variable_name_type_dimension (str): The variable name and type dimension of this group.
        complexity_candidate_list (list): A list of complexity candidates for this group.
        complexity_candidate (ComplexityCandidate): The selected complexity candidate for this group.
        sum_order (int): The sum of the orders of the complexities in this group.
        max_time (float): The maximum time of the complexities in this group.
        is_a_node (bool): Whether this group is a node or not.
        print_info (bool): Whether to print information during calculations.
    Methods:
        __init__: Initializes a new instance of the GroupComplexity class.
        add_group_or_node: Adds a group or node to this group.
        add_complexity: Adds a complexity to this group.
        remove_negligeable_complexity: Removes negligible complexity from this group.
        self_set_complexity: Sets the complexity of this group based on the given parameters.
        self_adjust_group_operations: Adjusts the group operations for this group.
        turn_into_a_node: Turns this group into a node.
        self_adjust_constant_complexity: Adjusts the constant complexity for this group.
        remove_constant_complexity: Removes constant complexity from this group.
        get_encapsulated_variable_name_type_dimension: Returns the encapsulated variable name and type dimension of this group.
        format_complexity: Formats the complexity of this group as a string.
    Examples:
        >>> group = GroupComplexity()
        >>> group.add_group_or_node(NodeComplexity('variable_name'))
        >>> group.self_set_complexity(max_time_rate=0.8, elect_function=np.argmin)
        >>> print(group.format_complexity(letter_list=['n', 'm', 'k'], next_letter_index=0))
        ('O(n)', 1, <function GroupComplexity.encapsulating_coeff_operation at 0x...>, None)
    """

    def __init__(self):
        self.group_or_node_list = []
        self.operation = None
        self.coeff_operation = None
        self.encapsulating_coeff_operation = None
        self.encapsulating_template = '{}'
        self.variable_name_type_dimension = None

        self.complexity_candidate_list = []
        self.complexity_candidate = None
        self.sum_order = None #sum_order of the subtree if is_a_node is False, else of the attached complexity
        self.max_time = None #max_time of the subtree if is_a_node is False, else of the attached complexity
        self.is_a_node = False

        self.print_info = False
        
    def add_group_or_node(self, group_or_node):
        self.group_or_node_list.append(group_or_node)

    def add_complexity(self, complexity_candidate):
        self.complexity_candidate_list.append(complexity_candidate)

    def remove_negligeable_complexity(self, max_time):
        """
        Propagates the absolute max_time in all the childrens, and if any of this children is a leaf, 
        then its complexity get to be updated if negligeable
        """
        if self.is_a_node:
            raise Exception('not implemented')

        else:
            if ((self.complexity_candidate.max_time / max_time) < 0.2) and (self.complexity_candidate.class_ != 'o(1)'):
                self.complexity_candidate.class_ = 'o(1)'
                self.complexity_candidate.coeff = self.complexity_candidate.max_time
                self.complexity_candidate.order = 0

            for group_or_node in self.group_or_node_list:
                group_or_node.remove_negligeable_complexity(max_time)

    def self_set_complexity(
        self, 
        max_time = None, 
        fix_negligeable_complexity = True,
        max_time_rate = 0.8, 
        elect_function = np.argmin, 
        fix_constant_complexity = True,
    ):
        """
        Set its complexity and the one of its children
        Also applies at the end the negligeable complexity function
        """
        if self.is_a_node:
            raise Exception('not implemented')

        # compute the complexity candidate of the grouped complexities
        # will be O(1) if there are no grouped complexities
        self.complexity_candidate = compute_complexity_candidate(
            complexity_candidate_list = self.complexity_candidate_list, 
            max_time_rate = max_time_rate, 
            elect_function = elect_function, 
            use_backup = False
        )

        # compute the complexity of each of the subelements
        self.max_time = 0
        for group_or_node in self.group_or_node_list:
            group_or_node.self_set_complexity(
                max_time = None, 
                fix_negligeable_complexity = False,
                max_time_rate = max_time_rate,
                elect_function = elect_function, 
                fix_constant_complexity = False,
            )

            if self.print_info:
                pass

            if group_or_node.max_time > self.max_time:
                self.max_time = group_or_node.max_time
                self.variable_name_type_dimension = group_or_node.variable_name_type_dimension

        if self.print_info:
            pass
        
        # fix negligeable complexity with the tracked time
        if fix_negligeable_complexity:
            if max_time is None:
                max_time = self.max_time

            if max_time > 0:
                self.remove_negligeable_complexity(
                    max_time = max_time, 
                )

    def self_adjust_group_operations(
        self, 
        max_time = None, 
        fix_negligeable_complexity = True,
        max_time_rate = 0.8, 
        elect_function = np.argmin, 
        fix_constant_complexity = True,
    ):
        """
        Should be apply after the tree after its set complexities
        Every node that has children (so group node) need to adjust its inner operation
        """
        for group_or_node in self.group_or_node_list:
            group_or_node.self_adjust_group_operations(
                max_time = None, 
                fix_negligeable_complexity = False,
                max_time_rate = max_time_rate,
                elect_function = elect_function, 
                fix_constant_complexity = False,
            )

        # fix the sum order
        sum_order_list = [group_or_node.sum_order for group_or_node in self.group_or_node_list]
        self.sum_order = sum(sum_order_list)


        sum_order_wolog_list = [int(group_or_node.sum_order) for group_or_node in self.group_or_node_list]
        sum_order_wolog = sum(sum_order_wolog_list)
        complexity_candidate_order_wolog = int(self.complexity_candidate.order)

        if sum_order_wolog == 0 and complexity_candidate_order_wolog > 0:
            # we turn this group complexity into a node (and discard the subtree)
            if self.print_info:
                print('turning into a node', sum_order_list, self.complexity_candidate.order)
            self.turn_into_a_node()

        elif (
            sum_order_wolog == complexity_candidate_order_wolog
        ):
            if self.sum_order < self.complexity_candidate.order:
                # the diff is a log
                self.operation = '*'
                self.coeff_operation = (lambda u_1, u_2: u_1 * u_2)
                self.encapsulating_coeff_operation = (lambda u_1: (u_1 * np.log(u_1)) if u_1 > 0 else 0)
                self.encapsulating_template = '{}log{}'
                self.sum_order = self.complexity_candidate.order

            else:
                self.operation = '*'
                self.coeff_operation = (lambda u_1, u_2: u_1 * u_2)
                self.encapsulating_coeff_operation = (lambda u_1: u_1)     
            
            
        elif (
            (sum_order_wolog > 0) and (complexity_candidate_order_wolog == 0)
        ):
            self.operation = '*'
            self.coeff_operation = (lambda u_1, u_2: u_1 * u_2)
            self.encapsulating_coeff_operation = (lambda u_1: u_1)

        elif all([temp_order < complexity_candidate_order_wolog for temp_order in sum_order_wolog_list]):
            # we need to adjust
            self.operation = '*'
            self.coeff_operation = (lambda u_1, u_2: u_1 * u_2)

            if sum_order_wolog >= complexity_candidate_order_wolog:
                self.encapsulating_coeff_operation = (lambda u_1: u_1)

            else:
                if self.print_info:
                    print('adjusting....')
                    print(sum_order_list)
                    print(self.sum_order)
                    print(self.complexity_candidate.order)

                # we need to adjust

                if sum_order_wolog * 2 == complexity_candidate_order_wolog:
                    self.encapsulating_template = '({}**2)' 
                    self.encapsulating_coeff_operation = (lambda u_1: u_1 **2)
                    self.sum_order = self.sum_order * 2
                    
                else:
                    self.encapsulating_template = '{}'
                    self.encapsulating_coeff_operation = (lambda u_1: u_1)
                    self.sum_order = self.sum_order
                    self.turn_into_a_node()

        else:
            if all([temp_order < self.complexity_candidate.order for temp_order in sum_order_list]):
                # the diff is a log
                self.operation = '+'
                self.coeff_operation = (lambda u_1, u_2: max(u_1, u_2))
                self.encapsulating_coeff_operation = (lambda u_1: (u_1 * np.log(u_1)) if u_1 > 0 else 0)
                self.encapsulating_template = '{}log{}'
                self.sum_order = self.complexity_candidate.order

            else:
                self.operation = '+'
                self.coeff_operation = (lambda u_1, u_2: max(u_1, u_2))
                self.encapsulating_coeff_operation = (lambda u_1: u_1)     
                self.sum_order = max(sum_order_list)

    def turn_into_a_node(self):
        self.is_a_node = True
        self.self_set_complexity = (lambda max_time, fix_negligeable_complexity, max_time_rate, elect_function, fix_constant_complexity: NodeComplexity.self_set_complexity(
            self, 
            max_time = max_time, 
            fix_negligeable_complexity = fix_negligeable_complexity,
            max_time_rate = max_time_rate, 
            elect_function = elect_function, 
            fix_constant_complexity = fix_constant_complexity,
        ))
        self.remove_negligeable_complexity = (lambda max_time: NodeComplexity.remove_negligeable_complexity(self, max_time=max_time))
        self.remove_constant_complexity = (lambda max_time_rate, elect_function: NodeComplexity.remove_constant_complexity(
            self, max_time_rate=max_time_rate, elect_function=elect_function
        ))
        self.format_complexity = (lambda letter_list, next_letter_index: NodeComplexity.format_complexity(
            self, letter_list=letter_list, next_letter_index=next_letter_index,
        ))
        self.self_adjust_group_operations = (lambda max_time, fix_negligeable_complexity, max_time_rate, elect_function, fix_constant_complexity: NodeComplexity.self_adjust_group_operations(
            self,
            max_time=max_time, 
            fix_negligeable_complexity=fix_negligeable_complexity, 
            max_time_rate=max_time_rate, 
            elect_function=elect_function, 
            fix_constant_complexity=fix_constant_complexity,
        ))
        self.self_adjust_constant_complexity = (lambda max_time, fix_negligeable_complexity, max_time_rate, elect_function, fix_constant_complexity: NodeComplexity.self_adjust_constant_complexity(
            self,
            max_time=max_time, 
            fix_negligeable_complexity=fix_negligeable_complexity, 
            max_time_rate=max_time_rate, 
            elect_function=elect_function, 
            fix_constant_complexity=fix_constant_complexity,
        ))
        self.sum_order = self.complexity_candidate.order
        self.max_time = self.complexity_candidate.max_time
        self.operation = None
        self.coeff_operation = None
        self.encapsulating_coeff_operation = None

    def self_adjust_constant_complexity(
        self, 
        max_time = None, 
        fix_negligeable_complexity = True,
        max_time_rate = 0.8, 
        elect_function = np.argmin, 
        fix_constant_complexity = True,
    ):
        if fix_constant_complexity and self.sum_order == 0:
            self.remove_constant_complexity(max_time_rate, elect_function)

    def remove_constant_complexity(self, max_time_rate, elect_function):
        if self.is_a_node:
            raise Exception('not implemented')
        
        if self.print_info:
            pass

        for group_or_node in self.group_or_node_list:
            if (
                group_or_node.variable_name_type_dimension == self.variable_name_type_dimension
            ):
                group_or_node.remove_constant_complexity(max_time_rate, elect_function)
                break

        sum_order_list = [group_or_node.sum_order for group_or_node in self.group_or_node_list]
        self.sum_order = sum(sum_order_list)

    def get_encapsulated_variable_name_type_dimension(self):        
        return [group_or_node.get_encapsulated_variable_name_type_dimension() for group_or_node in self.group_or_node_list]

    def format_complexity(
        self, 
        letter_list = ['n', 'm', 'k', 'l', 'u', 'v', 'w'], 
        next_letter_index = 0,
    ):
        if self.is_a_node:
            raise Exception('not implemented')

        formatted_complexity_list = []
        coeff_product_list = []
        coeff_max = None

        if len(self.group_or_node_list) == 0:
            raise Exception('not children')

        for i, group_or_node in enumerate(self.group_or_node_list):
            formatted_complexity, next_letter_index, coeff_product_temp, coeff_max_temp = group_or_node.format_complexity(letter_list, next_letter_index)

            coeff_product_list.append(coeff_product_temp)

            if (
                group_or_node.variable_name_type_dimension == self.variable_name_type_dimension
            ):
                coeff_max = coeff_max_temp

            if formatted_complexity != '1':
                formatted_complexity_list.append(formatted_complexity)

        if self.print_info:
            print(self.encapsulating_template)
            print(self.complexity_candidate.class_, formatted_complexity_list)

        if len(formatted_complexity_list) == 0:
            return '1', next_letter_index, self.encapsulating_coeff_operation(reduce(self.coeff_operation, coeff_product_list)), coeff_max
        
        if self.print_info:
            pass
            
        complexity_str = (
            ('(' if len(formatted_complexity_list) > 1 else '')
            + 
            (self.operation.join(formatted_complexity_list))
            +
            (')' if len(formatted_complexity_list) > 1 else '')
        )

        if self.print_info:
            pass

        complexity_str = self.encapsulating_template.format(complexity_str, complexity_str, complexity_str, complexity_str)

        return_values = complexity_str, next_letter_index, self.encapsulating_coeff_operation(reduce(self.coeff_operation, coeff_product_list)), coeff_max

        if self.print_info:
            print('complexity:', self.get_encapsulated_variable_name_type_dimension(), return_values[0])

        return return_values

def compute_complexity_candidate(
    complexity_candidate_list, 
    max_time_rate, 
    elect_function, 
    use_backup=False
):
    """
    Computes the complexity candidate from a list of complexity candidates.
    Args:
        complexity_candidate_list (list): A list of ComplexityCandidate objects.
        max_time_rate (float): The maximum time rate to consider when computing the complexity candidate.
        elect_function (function): The function to use to elect the complexity candidate.
        use_backup (bool, optional): Whether to use the backup complexity candidate. Defaults to False.
    Returns:
        ComplexityCandidate: The computed complexity candidate.
    Notes:
        This function first filters the complexity candidates by priority and then by maximum time.
        It then uses the elect_function to select the complexity candidate with the highest order.
        If use_backup is True, it uses the backup complexity candidate instead of the main one.
    Examples:
        >>> complexity_candidates = [ComplexityCandidate(class_='O(n)', order=1), ComplexityCandidate(class_='O(n^2)', order=2)]
        >>> compute_complexity_candidate(complexity_candidates, 0.8, np.argmin)
        ComplexityCandidate(class_='O(n^2)', order=2, ...)
    """
    
    if len(complexity_candidate_list) == 0:
        return ComplexityCandidate()

    priority_list = list(map(lambda x: x.priority, complexity_candidate_list))
    complexity_candidate_per_priority_list = []

    for priority in priority_list:
        complexity_candidate_list_per_priority = list(filter(lambda x: x.priority == priority, complexity_candidate_list))

        max_time = max(map(lambda x: x.max_time, complexity_candidate_list_per_priority))
            
        complexity_candidate_list_per_priority = list(filter(
            lambda x: x.max_time >= max_time * max_time_rate, 
            complexity_candidate_list_per_priority
        ))

        complexity_candidate_per_priority_list.append(complexity_candidate_list_per_priority[elect_function(list(map(
            lambda x: (x.order if not use_backup else x.order_backup), 
            complexity_candidate_list_per_priority
        )))])

    complexity_candidate = complexity_candidate_per_priority_list[np.argmax(list(map(
        lambda x: (x.order if not use_backup else x.order_backup), 
        complexity_candidate_per_priority_list
    )))]

    return ComplexityCandidate(        
        class_ = (complexity_candidate.class_ if not use_backup else complexity_candidate.class_backup), 
        order = (complexity_candidate.order if not use_backup else complexity_candidate.order_backup),
        class_backup = None,
        order_backup = None,
        max_time = complexity_candidate.max_time,
        priority = complexity_candidate.priority,
        found_peak = complexity_candidate.found_peak,
        coeff = (complexity_candidate.coeff if not use_backup else complexity_candidate.coeff_backup), 
        coeff_backup = None, 
    )