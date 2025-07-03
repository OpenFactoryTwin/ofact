"""
Sort the process_executions to ensure the right execution
Mainly difficult because some 'logical' process_executions can take place at the same time
Therefore a chronological order is needed but only but does not lead to the right execution order alone
"""


def _get_sorted_process_executions(process_executions, tuple_=False):
    """Return sorted process_executions
    How to sort the process_executions:
    1. according to the executed_end_time
    2. according to the negative duration to ensure the longer process_executions should be executed first
    (with the same end_time)
    3. destination - origin relation for two process_executions with the same start_time and end_time
    """

    # sort according a process_set processes = set(processes)
    if not tuple_:
        sorted_process_executions = \
            sorted(process_executions,
                   key=lambda process_execution: (process_execution.executed_end_time,
                                                  - (process_execution.executed_end_time -
                                                     process_execution.executed_start_time)),
                   reverse=False)

        has_same = has_same_attribute(sorted_process_executions)
        if has_same:
            if not sorted_process_executions[has_same].origin == sorted_process_executions[has_same - 1].destination:
                # switching needed
                first = sorted_process_executions[has_same]
                second = sorted_process_executions[has_same - 1]
                lst = [first, second]
                sorted_process_executions[has_same - 1: has_same + 1] = lst

                # print("Process Sequence: ",
                #       [(process_execution.resourced_used, process_execution.process.name)
                #        for process_execution in sorted_process_executions])

    else:
        sorted_process_executions = \
            sorted(process_executions,
                   key=lambda process_execution_with_provider:
                   (process_execution_with_provider[0].executed_end_time,
                    - (process_execution_with_provider[0].executed_end_time -
                       process_execution_with_provider[0].executed_start_time)),
                   reverse=False)

        has_same = has_same_attribute(sorted_process_executions, tuple_=True)
        if has_same:
            if not sorted_process_executions[has_same][0].origin == \
                   sorted_process_executions[has_same - 1][0].destination:
                # switching needed
                first = sorted_process_executions[has_same]
                second = sorted_process_executions[has_same - 1]
                lst = [first, second]
                sorted_process_executions[has_same - 1: has_same + 1] = lst

                # print("Process Sequence: ",
                #       [(process_execution[0].resourced_used, process_execution[0].process.name)
                #        for process_execution in sorted_process_executions])

    return sorted_process_executions


def has_same_attribute(lst, tuple_=False):
    if len(lst) < 2:
        return False

    for i in range(1, len(lst)):
        if not tuple_:
            predecessor = lst[i - 1].__dict__["_executed_end_time"] - lst[i - 1].__dict__["_executed_start_time"]
            successor = lst[i].__dict__["_executed_end_time"] - lst[i].__dict__["_executed_start_time"]

        else:
            predecessor = lst[i - 1][0].__dict__["_executed_end_time"] - lst[i - 1][0].__dict__["_executed_start_time"]
            successor = lst[i][0].__dict__["_executed_end_time"] - lst[i][0].__dict__["_executed_start_time"]

        if predecessor == successor and predecessor.seconds == 0:
            return i

    return False
