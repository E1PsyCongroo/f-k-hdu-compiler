from lab2 import *

def main():
    patterns = ["ab", "a|c", "a(b|c)", "(a|b)*", "(a|b)+", "ab+c?", "(ab)*|c+", "b(a|b)*aa", "(a|b)*abb"]
    strings = ["ab", "abc", "abcc", "c", "abbbbb", "abab", "bababababaaa"]

    for pattern in patterns:
        regex = Regex(pattern)
        print(f"\nTesting pattern: {regex.pattern}")
        nfa = regex.to_nfa()
        dfa = DFA(nfa)
        mini_dfa = dfa.minimize()
        nfa.visualize('result/nfa_' + pattern)
        dfa.visualize('result/dfa_' + pattern)
        mini_dfa.visualize('result/dfa_minimize_' + pattern)
        for string in strings:
            nfa_result = nfa.simulate(string)
            dfa_result = dfa.simulate(string)
            mini_dfa_result = mini_dfa.simulate(string)
            assert(nfa_result == dfa_result)
            assert(mini_dfa_result == dfa_result)
            if (nfa_result):
                print(f"\033[32mMatched: {string}\033[0m")
            else:
                print(f"\033[31mUnmatched: {string}\033[0m")

if __name__ == "__main__":
    main()
