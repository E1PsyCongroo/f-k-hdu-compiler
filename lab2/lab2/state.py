class State:
    id_counter = 0

    def __init__(self, name=None):
        self.name = name or f"S{State.id_counter}"
        State.id_counter += 1
        self.transitions: dict[str, list[State]] = {}

    def add_transition(self, input_char, state):
        self.transitions.setdefault(input_char, []).append(state)

    def __str__(self):
        return self.name
