from .trie import *


class CFG:
    def __init__(self, read=False):
        self.terminalSyms: set[str] = set()  # 终结符号集
        self.startSym: str = None  # 开始符号
        self.grammar: dict[str, list[list[str]]] = {}  # 产生式

        if read:
            self.read_grammar()

    def read_grammar(self) -> None:
        symbal: set[str] = set()
        while True:
            try:
                self.startSym = input("请输入文法开始符号: ").strip()  # 获取开始符号
                assert len(self.startSym) == 1
                break
            except AssertionError:
                print("\033[31m上下文无关语法有且仅有一个开始符号\033[0m")
        symbal.add(self.startSym)
        print(f"\033[32m已读取开始符号: '{self.startSym}'\033[0m")

        print("请输入文法（使用 'END' 来结束输入）: ")
        while True:
            try:
                line = input()
            except EOFError:
                break

            if line.strip().upper() == "END":
                break
            if "->" in line:
                non_terminal, productions = line.split("->")
                non_terminal = non_terminal.strip()
                try:
                    assert len(self.startSym) == 1
                except AssertionError:
                    print("\033[31m\033上下文无关语法产生式左侧只能有一个非总结符号[0m")
                productions = [prod.split() for prod in productions.split("|")]
                self.add_rule(non_terminal, productions)

    def set_start(self, startSym: str) -> None:
        self.startSym = startSym

    def add_rule(self, nonterminalSym: str, productions: list[list[str]]) -> None:
        if nonterminalSym not in self.grammar:
            self.grammar[nonterminalSym] = productions
        else:
            self.grammar[nonterminalSym] += productions

        if nonterminalSym in self.terminalSyms:
            self.terminalSyms.remove(nonterminalSym)
        self.terminalSyms.update(
            sym
            for production in productions
            for sym in production
            if sym not in self.grammar
        )

    def eliminate_left_recursion(self) -> None:
        # 非终结符号集
        nonterminalSyms = list(self.grammar.keys())

        for i in range(len(nonterminalSyms)):
            nonterminalSym = nonterminalSyms[i]
            productions = self.grammar[nonterminalSym]

            # 非递归右部
            nonrecursiveProductions: list[list[str]] = []
            # 递归右部
            recursiveProductions: list[list[str]] = []

            # 生成间接右部
            for j in range(i):
                for prod in productions.copy():
                    if prod[0] == nonterminalSyms[j]:  # 间接右部生成
                        productions.remove(prod)
                        productions.extend(
                            [
                                prodj + prod[1:]
                                for prodj in self.grammar[nonterminalSyms[j]]
                            ]
                        )

            for newProd in productions:
                if newProd[0] == nonterminalSym:
                    recursiveProductions.append(newProd[1:])
                else:
                    nonrecursiveProductions.append(newProd)

            # 处理直接左递归
            if recursiveProductions:
                # 新非终结符号
                newNonterminalSym = f"{nonterminalSym}'"
                for prod in nonrecursiveProductions:
                    prod.append(newNonterminalSym)
                self.grammar[nonterminalSym] = nonrecursiveProductions
                for prod in recursiveProductions:
                    prod.append(newNonterminalSym)
                recursiveProductions.append(["ε"])
                self.grammar[newNonterminalSym] = recursiveProductions
            else:
                self.grammar[nonterminalSym] = nonrecursiveProductions

    def extract_left_common_factors(self):
        newGrammar: dict[str, list[list[str]]] = {key: [] for key in self.grammar}

        for nonterminalSym, productions in self.grammar.items():
            # 如果产生式右部符号个数≤1, 直接赋值
            if len(productions) <= 1:
                newGrammar[nonterminalSym] = productions
                continue

            # 将产生式插入到 Trie 中
            trie = Trie(nonterminalSym)
            for prod in productions:
                trie.insert(prod)

            # 获取最长公共因子式
            commonPrefixes = trie.get_prefixes()

            if commonPrefixes:
                newNonterminalSym = nonterminalSym
                prefixMap: list[tuple[list[str], str]] = []
                for commonPrefix in commonPrefixes:
                    newNonterminalSym = f"{newNonterminalSym}'"
                    prefixMap.append((commonPrefix, newNonterminalSym))
                    newGrammar[newNonterminalSym] = []
                    newGrammar[nonterminalSym].append(
                        commonPrefix + [newNonterminalSym]
                    )

                for prod in productions:
                    for commonPrefix, newNonterminalSym in prefixMap:
                        commonPrefixLen = len(commonPrefix)
                        if prod[:commonPrefixLen] == commonPrefix:
                            newGrammar[newNonterminalSym].append(prod[commonPrefixLen:])
                            break
                    else:
                        newGrammar[nonterminalSym].append(prod)
            else:
                newGrammar[nonterminalSym] = productions

        self.grammar = newGrammar

    def compute_firstSets(self) -> dict[str, set[str]]:
        firstSets: dict[str, set[str]] = {}
        # 初始化所有非终结符的 FIRST 集
        for nonterminalSym in self.grammar.keys():
            firstSets[nonterminalSym] = set()

        def compute_first(symbol: str) -> set[str]:
            if symbol not in firstSets:
                return {symbol}
            if firstSets[symbol]:
                return firstSets[symbol]
            for prod in self.grammar[symbol]:
                count = 0
                for prodSym in prod:
                    symFirst = compute_first(prodSym)
                    if "ε" not in symFirst:
                        firstSets[symbol].update(symFirst)
                        break
                    count += 1
                    firstSets[symbol].update(symFirst - {"ε"})
                if count == len(prod):
                    firstSets[symbol].add("ε")

            return firstSets[symbol]

        for nonterminalSym in firstSets.keys():
            compute_first(nonterminalSym)

        return firstSets

    def compute_followSets(self) -> dict[str, set[str]]:
        # 获取 FIRST 集
        firstSets = self.compute_firstSets()

        followSets: dict[str, set[str]] = {}
        # 初始化所有非终结符的 FOLLOW 集
        for nonterminal in self.grammar.keys():
            followSets[nonterminal] = set()
        followSets[self.startSym].add("$")  # $ 为输入的结束符

        # 迭代直到所有 FOLLOW 集不再变化
        changed = True
        while changed:
            changed = False
            for nonterminal, productions in self.grammar.items():
                for production in productions:
                    for i, symbol in enumerate(production):
                        if symbol in firstSets:  # 当前符号是非终结符
                            originalSize = len(followSets[symbol])

                            if i + 1 < len(production):  # 右侧还有符号
                                nextSymbol = production[i + 1]

                                if nextSymbol in firstSets:
                                    followSets[symbol].update(firstSets[nextSymbol] - {"ε"})
                                    if "ε" in firstSets[nextSymbol]:
                                        followSets[symbol].update(followSets[nonterminal])
                                else:
                                    followSets[symbol].add(nextSymbol)
                            else:  # 如果是最后一个符号，添加非终结符的 FOLLOW 集
                                followSets[symbol].update(followSets[nonterminal])

                            changed |= originalSize != len(followSets[symbol])

        return followSets

    def display(self):
        print(f"开始符号: '{self.startSym}'")
        print(f"非终结符号集: [{", ".join(self.grammar.keys())}]")
        print(f"终结符号集: [{", ".join(self.terminalSyms)}]")
        print("产生式:")
        for nonterminalSym in self.grammar:
            productions = " | ".join(
                [" ".join(production) for production in self.grammar[nonterminalSym]]
            )
            print(f"{nonterminalSym} -> {productions}")
