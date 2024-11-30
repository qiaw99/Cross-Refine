ECQA_GRAMMAR = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 | op2 | op3 | op4 | op5

op1: " 1"
op2: " 2"
op3: " 3"
op4: " 4"
op5: " 5"
"""

eSNLI_grammar = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 | op2 | op3 

op1: " entailment"
op2: " neutral"
op3: " contradiction"
"""

healthFC_grammar = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 | op2 | op3

op1: " yes"
op2: " no"
op3: " unknown"
"""


improvement_grammar = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 | op2 

op1: " True"
op2: " False"
"""
