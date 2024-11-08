from lab2 import *

def main():
    patterns = ["ab", "a|c", "a(b|c)", "(a|b)*", "(a|b)+", "ab+c?", "(ab)*|c+"]
    strings = ["ab", "abc", "abcc", "c", "abbbbb", "abab"]

    for pattern in patterns:
        regex = Regex(pattern)
        print(f"\nTesting pattern: {regex.pattern}")
        print(regex.to_postfix())
        nfa = regex.to_nfa()
        dfa = DFA(nfa)
        dfa.visualize('dfa_' + pattern)
        dfa.minimize()
        dfa.visualize('dfa_minimize_' + pattern)
        nfa.visualize('nfa_' + pattern)
        for string in strings:
            nfa_result = nfa.simulate(string)
            dfa_result = dfa.simulate(string)
            assert(nfa_result == dfa_result)
            print(f"String: '{string}' Accepted: {dfa_result}")

if __name__ == "__main__":
    main()
