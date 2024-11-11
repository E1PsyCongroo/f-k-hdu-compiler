import pytest
from lab3 import *


def test_cfg_eliminate_left_recursion1():
    cfg = CFG(False)

    cfg.set_start("S")
    cfg.add_rule("S", [["S", "+", "T"], ["T"]])
    cfg.add_rule("T", [["T", "*", "F"], ["F"]])
    cfg.add_rule("F", [["(", "E", ")"], ["id"]])

    print("原始文法:")
    cfg.display()

    cfg.eliminate_left_recursion()

    print("消去左递归后的文法:")
    cfg.display()

    expected_rules: dict[str, list[list[str]]] = {
        "S": [["T", "S'"]],
        "S'": [["+", "T", "S'"], ["ε"]],
        "T": [["F", "T'"]],
        "T'": [["*", "F", "T'"], ["ε"]],
        "F": [["(", "E", ")"], ["id"]],
    }

    assert sorted(cfg.grammar.keys()) == sorted(expected_rules.keys())
    for non_terminal, productions in expected_rules.items():
        assert sorted(["".join(prod) for prod in cfg.grammar[non_terminal]]) == sorted(
            ["".join(prod) for prod in productions]
        )


def test_cfg_eliminate_left_recursion2():
    cfg = CFG(False)

    cfg.set_start("S")
    cfg.add_rule("S", [["A", "c"], ["c"]])
    cfg.add_rule("A", [["B", "b"], ["b"]])
    cfg.add_rule("B", [["S", "a"], ["a"]])

    print("原始文法:")
    cfg.display()

    cfg.eliminate_left_recursion()

    print("消去左递归后的文法:")
    cfg.display()

    expected_rules: dict[str, list[list[str]]] = {
        "S": [["A", "c"], ["c"]],
        "A": [["B", "b"], ["b"]],
        "B": [["b", "c", "a", "B'"], ["c", "a", "B'"], ["a", "B'"]],
        "B'": [["b", "c", "a", "B'"], ["ε"]],
    }

    assert sorted(cfg.grammar.keys()) == sorted(expected_rules.keys())
    for non_terminal, productions in expected_rules.items():
        assert sorted(["".join(prod) for prod in cfg.grammar[non_terminal]]) == sorted(
            ["".join(prod) for prod in productions]
        )


def test_extract_left_common_factors():
    cfg = CFG(False)

    cfg.set_start("S")
    cfg.add_rule(
        "S",
        [
            list("apple"),
            list("apply"),
            list("application"),
            list("ball"),
            list("bat"),
            list("bath"),
            list("Xb"),
        ],
    )
    cfg.add_rule("X", [list("ab"), list("ac"), list("ad")])

    print("原始文法:")
    cfg.display()

    cfg.extract_left_common_factors()

    print("消去公因式后的文法:")
    cfg.display()

    expected_rules: dict[str, list[list[str]]] = {
        "S": [list("appl") + ["S'"], ["b", "a", "S''"], ["X", "b"]],
        "S'": [["e"], ["y"], list("ication")],
        "S''": [["l", "l"], ["t"], ["t", "h"]],
        "X": [["a", "X'"]],
        "X'": [["b"], ["c"], ["d"]],
    }

    assert sorted(cfg.grammar.keys()) == sorted(expected_rules.keys())
    for non_terminal, productions in expected_rules.items():
        assert sorted(["".join(prod) for prod in cfg.grammar[non_terminal]]) == sorted(
            ["".join(prod) for prod in productions]
        )


def test_compute_first1():
    cfg = CFG(False)

    cfg.set_start("S")
    cfg.add_rule("S", [["A", "B"]])
    cfg.add_rule("A", [["a"], ["ε"]])
    cfg.add_rule("B", [["b"]])

    print("文法:")
    cfg.display()

    firstSets = cfg.compute_firstSets()
    print("FIRST 集:", firstSets)

    expectedFirstSets: dict[str, set[str]] = {
        "S": {"a", "b"},
        "A": {"a", "ε"},
        "B": {"b"},
    }

    assert len(firstSets) == len(expectedFirstSets)
    for nonterminalSym in expectedFirstSets.keys():
        assert firstSets[nonterminalSym] == expectedFirstSets[nonterminalSym]


def test_compute_first2():
    cfg = CFG(False)

    cfg.set_start("E")
    cfg.add_rule("E", [["T", "X"]])
    cfg.add_rule("T", [["int", "Y"], ["(", "E", ")"]])
    cfg.add_rule("X", [["+", "E"], ["ε"]])
    cfg.add_rule("Y", [["*", "T"], ["ε"]])

    print("文法:")
    cfg.display()

    firstSets = cfg.compute_firstSets()
    print("FIRST 集:", firstSets)

    expectedFirstSets: dict[str, set[str]] = {
        "E": {"(", "int"},
        "T": {"(", "int"},
        "X": {"+", "ε"},
        "Y": {"*", "ε"},
    }

    assert len(firstSets) == len(expectedFirstSets)
    for nonterminalSym in expectedFirstSets.keys():
        assert firstSets[nonterminalSym] == expectedFirstSets[nonterminalSym]


def test_compute_follow1():
    cfg = CFG(False)

    cfg.set_start("S")
    cfg.add_rule("S", [["A", "B"]])
    cfg.add_rule("A", [["a"], ["ε"]])
    cfg.add_rule("B", [["b"]])

    print("文法:")
    cfg.display()

    followSets = cfg.compute_followSets()
    print("FOLLOW 集:", followSets)

    expectedFollowSets: dict[str, set[str]] = {
        "S": {"$"},
        "A": {"b"},
        "B": {"$"},
    }

    assert len(followSets) == len(expectedFollowSets)
    for nonterminalSym in expectedFollowSets.keys():
        assert followSets[nonterminalSym] == expectedFollowSets[nonterminalSym]


def test_compute_follow2():
    cfg = CFG(False)

    cfg.set_start("E")
    cfg.add_rule("E", [["T", "X"]])
    cfg.add_rule("T", [["int", "Y"], ["(", "E", ")"]])
    cfg.add_rule("X", [["+", "E"], ["ε"]])
    cfg.add_rule("Y", [["*", "T"], ["ε"]])

    print("文法:")
    cfg.display()

    followSets = cfg.compute_followSets()
    print("FOLLOW 集:", followSets)

    expectedFollowSets: dict[str, set[str]] = {
        "E": {"$", ")"},
        "X": {"$", ")"},
        "T": {"+", "$", ")"},
        "Y": {"+", "$", ")"},
    }

    assert len(followSets) == len(expectedFollowSets)
    for nonterminalSym in expectedFollowSets.keys():
        assert followSets[nonterminalSym] == expectedFollowSets[nonterminalSym]


def test_compute_follow3():
    cfg = CFG(False)

    cfg.set_start("E")
    cfg.add_rule("E", [["T", "E'"]])
    cfg.add_rule("E'", [["+", "T", "E'"], ["ε"]])
    cfg.add_rule("T", [["F", "T'"]])
    cfg.add_rule("T'", [["*", "F", "T'"], ["ε"]])
    cfg.add_rule("F", [["(", "E", ")"], ["id"]])

    print("文法:")
    cfg.display()

    followSets = cfg.compute_followSets()
    print("FOLLOW 集:", followSets)

    expectedFollowSets: dict[str, set[str]] = {
        "E": {"$", ")"},
        "E'": {"$", ")"},
        "T": {"+", "$", ")"},
        "T'": {"+", "$", ")"},
        "F": {"*", "+", "$", ")"},
    }

    assert len(followSets) == len(expectedFollowSets)
    for nonterminalSym in expectedFollowSets.keys():
        assert followSets[nonterminalSym] == expectedFollowSets[nonterminalSym]
