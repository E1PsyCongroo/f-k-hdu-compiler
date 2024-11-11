from lab2.nfa import *
from lab2.state import *


class Regex:
    def __init__(self, pattern: str):
        def add_explicit_concat_operator(expression: str):
            """添加显示连接符号"""
            output = []
            unaryOperators = {"*", "+", "?"}
            binaryOperators = {"|"}
            operators = unaryOperators.union(binaryOperators)
            for i in range(len(expression) - 1):
                output.append(expression[i])
                assert not (
                    expression[i] in unaryOperators
                    and expression[i + 1] in unaryOperators
                )
                assert not (
                    expression[i] in binaryOperators
                    and expression[i + 1] in binaryOperators
                )
                assert not (
                    expression[i] in binaryOperators
                    and expression[i + 1] in unaryOperators
                )
                if (expression[i] not in binaryOperators and expression[i] != "(") and (
                    expression[i + 1] not in operators and expression[i + 1] != ")"
                ):
                    output.append(".")
            output.append(expression[-1])
            return "".join(output)

        self.pattern: str = add_explicit_concat_operator(pattern)

    def to_postfix(self):
        """将正则表达式转换为后缀表达式"""
        precedence = {".": 2, "|": 1}  # '.'表示连接操作
        output = []
        stack = []
        for char in self.pattern:
            if char in precedence:
                while (
                    stack
                    and stack[-1] != "("
                    and precedence[stack[-1]] >= precedence[char]
                ):
                    output.append(stack.pop())
                stack.append(char)
            elif char == "(":
                stack.append(char)
            elif char == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                stack.pop()
            else:
                output.append(char)
        while stack:
            output.append(stack.pop())
        return "".join(output)

    def to_nfa(self):
        postfix = self.to_postfix()
        stack: list[NFA] = []
        for char in postfix:
            if char == "*":
                stack.append(self.apply_closure(stack.pop()))
            elif char == "+":
                stack.append(self.apply_plus(stack.pop()))
            elif char == "?":
                stack.append(self.apply_question(stack.pop()))
            elif char == "|":
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                stack.append(self.apply_union(nfa1, nfa2))
            elif char == ".":
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                stack.append(self.apply_concatenation(nfa1, nfa2))
            else:
                stack.append(self.create_basic_nfa(char))
        return stack.pop()

    def create_basic_nfa(self, char):
        """实现单个符号的NFA"""
        start_state = State()
        accept_state = State()
        start_state.add_transition(char, accept_state)
        return NFA([start_state, accept_state])

    def apply_closure(self, nfa: NFA):
        """实现闭包操作: NFA*"""
        start_state = State()
        accept_state = State()
        start_state.add_transition(None, nfa.start_state)
        start_state.add_transition(None, accept_state)
        nfa.accept_state.add_transition(None, nfa.start_state)
        nfa.accept_state.add_transition(None, accept_state)
        nfa.add_state(start_state)
        nfa.add_state(accept_state)
        nfa.set_start_state(start_state)
        nfa.set_accept_state(accept_state)
        return nfa

    def apply_plus(self, nfa: NFA):
        """实现一次或多次重复: NFA+ 相当于 NFA . NFA*"""
        nfa_copy = nfa.copy()
        nfa_copy.visualize("test")
        return self.apply_concatenation(nfa_copy, self.apply_closure(nfa))

    def apply_question(self, nfa: NFA):
        """实现零次或一次出现: NFA? 相当于 (ε|NFA)"""
        start_state = State()
        accept_state = State()
        start_state.add_transition(None, accept_state)
        epsilon_nfa = NFA([start_state, accept_state])
        return self.apply_union(epsilon_nfa, nfa)

    def apply_union(self, nfa1: NFA, nfa2: NFA):
        """实现并联操作: NFA1 | NFA2"""
        start_state = State()
        accept_state = State()
        start_state.add_transition(None, nfa1.start_state)
        start_state.add_transition(None, nfa2.start_state)
        nfa1.accept_state.add_transition(None, accept_state)
        nfa2.accept_state.add_transition(None, accept_state)
        return NFA([start_state] + nfa1.states + nfa2.states + [accept_state])

    def apply_concatenation(self, nfa1: NFA, nfa2: NFA):
        """实现连接操作: NFA1 . NFA2"""
        nfa1.accept_state.add_transition(None, nfa2.start_state)
        nfa = NFA(nfa1.states + nfa2.states)
        nfa.set_start_state(nfa1.start_state)
        nfa.set_accept_state(nfa2.accept_state)
        return nfa
