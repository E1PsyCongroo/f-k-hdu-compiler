from lab3 import *

# 测试程序
def main():
    # 从输入读取文法
    cfg = CFG(read=True)

    print("\n原始文法:")
    cfg.display()

    cfg.eliminate_left_recursion()
    cfg.extract_left_common_factors()

    print("文法是否满足LL(1):", cfg.is_ll1())
    cfg.construct_predictive_table()

    print("\n处理后的文法:")
    cfg.display()

    while True:
        try:
            inputStr = input("请输入待分析串:").strip()
            if inputStr:
                print(f"\033[32m已获取输入串: '{inputStr}'\033[0m")
                print(f"分析结果: {cfg.parse(inputStr.split(" "))}")
        except EOFError:
            break

    print("程序结束")

if __name__ == "__main__":
    main()
