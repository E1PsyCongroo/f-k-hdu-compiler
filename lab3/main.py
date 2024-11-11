from lab3 import *

# 测试程序
def main():
    # 从输入读取文法
    cfg = CFG(read=True)

    print("原始文法:")
    cfg.display()

    # 消去左递归
    cfg.eliminate_left_recursion()

    print("消去左递归后的文法:")
    cfg.display()


if __name__ == "__main__":
    main()
