import graphviz
from lab2.nfa import *
from lab2.state import *

class DFA:
    def __init__(self, nfa: NFA=None):
        """初始化DFA。如果提供NFA, 将其转换为DFA"""
        self.states: list[State] = []
        self.start_state: State = None
        self.accept_states: list[State] = []
        if nfa:
            self.initialize_from_nfa(nfa)

    def initialize_from_nfa(self, nfa: NFA):
        """根据给定的NFA构建DFA"""
        initial_closure = nfa.epsilon_closure([nfa.start_state])
        start_state = State()
        self.add_state(start_state, nfa.start_state == nfa.accept_state)
        self.set_start_state(start_state)

        unmarked = [(start_state, initial_closure)]
        marked = {}
        marked[frozenset(initial_closure)] = start_state

        while unmarked:
            current_dfa_state, current_nfa_states = unmarked.pop(0)
            for input_char in set(char for state in current_nfa_states for char in state.transitions if char is not None):
                new_nfa_states = nfa.epsilon_closure(nfa.move(current_nfa_states, input_char))
                state_set = frozenset(new_nfa_states)
                if state_set not in marked:
                    new_dfa_state = State()
                    self.add_state(new_dfa_state, any(s == nfa.accept_state for s in new_nfa_states))
                    marked[state_set] = new_dfa_state
                    unmarked.append((new_dfa_state, new_nfa_states))
                current_dfa_state.add_transition(input_char, marked[state_set])

    def add_state(self, state: State, is_accept=False):
        """添加一个状态到DFA, 并标记是否为接受状态"""
        self.states.append(state)
        if is_accept:
            self.accept_states.append(state)

    def set_start_state(self, state: State):
        """设置DFA的起始状态"""
        self.start_state = state

    def simulate(self, input_string: str):
        """模拟DFA, 检查是否接受给定的输入字符串"""
        current_state = self.start_state
        for char in input_string:
            if char in current_state.transitions:
                current_state = current_state.transitions[char][0]
            else:
                return False
        return current_state in self.accept_states

    def visualize(self, filename:str):
        """使用Graphviz可视化DFA"""
        dot = graphviz.Digraph()
        dot.node("", shape="none")
        dot.edge("", str(self.start_state), label='start')
        for state in self.states:
            shape = 'doublecircle' if state in self.accept_states else 'circle'
            dot.node(str(state), shape=shape)
            for char, next_states in state.transitions.items():
                for next_state in next_states:
                    dot.edge(str(state), str(next_state), label=str(char))
        dot.render(filename, format="png", view=False)


    def minimize(self):
        """使用Hopcroft算法求异法最小化DFA"""
        symbols = {symbol for state in self.states for symbol in state.transitions}

        accept = frozenset(self.accept_states)
        non_accept = frozenset(self.states) - accept
        P = {accept, non_accept}
        W = P.copy()

        while W:
            A = W.pop()
            for symbol in symbols:
                X = {state for state in self.states if any(next_state in A for next_state in state.transitions.get(symbol, []))}
                if not X:
                    continue

                for Y in P.copy():
                    intersection = Y & X
                    difference = Y - X
                    if intersection and difference:
                        P.remove(Y)
                        P.add(frozenset(intersection))
                        P.add(frozenset(difference))
                        if Y in W:
                            W.remove(Y)
                            W.add(frozenset(intersection))
                            W.add(frozenset(difference))
                        else:
                            if len(intersection) <= len(difference):
                                W.add(frozenset(intersection))
                            else:
                                W.add(frozenset(difference))


        new_states = {frozenset(group): State() for group in P}
        new_start_state = next(new_states[group] for group in P if self.start_state in group)
        new_accept_states = {new_states[group] for group in P if any(s in accept for s in group)}

        for group, new_state in new_states.items():
            for state in group:
                for sym, destinations in state.transitions.items():
                    dest_group = next(g for g in P if any(d in g for d in destinations))
                    if new_states[dest_group] not in new_state.transitions.get(sym, []):
                        new_state.add_transition(sym, new_states[dest_group])

        new_dfa = DFA()
        new_dfa.states = list(new_states.values())
        new_dfa.start_state = new_start_state
        new_dfa.accept_states = list(new_accept_states)
        return new_dfa
