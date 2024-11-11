import graphviz
from lab2.state import *

class NFA:
    def __init__(self, states: list[State] = []):
        self.states: list[State] = states
        self.start_state: State = states[0] if states else None
        self.accept_state: State = states[-1] if states else None

    def add_state(self, state: State):
        """添加状态到NFA"""
        self.states.append(state)

    def set_start_state(self, state: State):
        """设置起始状态"""
        self.start_state = state

    def set_accept_state(self, state: State):
        """添加接受状态"""
        self.accept_state = state

    def epsilon_closure(self, states: list[State]) -> set[State]:
        """计算给定状态集的epsilon闭包"""
        stack = list(states)
        closure = set(stack)
        while stack:
            state = stack.pop()
            if None in state.transitions:
                for next_state in state.transitions[None]:
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)
        return closure

    def move(self, states: list[State], input_char) -> set[State]:
        """ 返回从给定状态集出发, 经过input_char可以到达的状态集 """
        next_states = set()
        for state in states:
            if input_char in state.transitions:
                next_states.update(state.transitions[input_char])
        return next_states

    def simulate(self, input_string: str):
        """模拟NFA处理输入字符串, 返回是否接受该字符串"""
        current_states = self.epsilon_closure([self.start_state])
        for char in input_string:
            next_states = set()
            for state in current_states:
                if char in state.transitions:
                    for next_state in state.transitions[char]:
                        next_states.update(self.epsilon_closure([next_state]))
            current_states = next_states
        return self.accept_state in current_states

    def visualize(self, filename: str):
        """可视化NFA, 使用Graphviz"""
        dot = graphviz.Digraph()
        dot.node("", shape="none")
        dot.edge("", str(self.start_state), label='start')
        states = [self.start_state]
        for state in self.states:
            shape = 'doublecircle' if state == self.accept_state else 'circle'
            dot.node(str(state), shape=shape)
            for char, states in state.transitions.items():
                label = char if char is not None else 'ε'
                for next_state in states:
                    dot.edge(str(state), str(next_state), label=label)
        dot.render(filename, format="png", view=False)

    def copy(self):
        """创建并返回此NFA的深拷贝"""
        state_map = {state: State() for state in self.states}
        new_nfa = NFA([])
        for state in self.states:
            new_state = state_map[state]
            new_nfa.add_state(new_state)
            for char, states in state.transitions.items():
                for target_state in states:
                    new_state.add_transition(char, state_map[target_state])
        if self.start_state:
            new_nfa.set_start_state(state_map[self.start_state])
        if self.accept_state:
            new_nfa.set_accept_state(state_map[self.accept_state])
        return new_nfa
