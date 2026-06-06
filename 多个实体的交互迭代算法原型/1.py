
def f1():
    W1 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    return W1

def f2():
    W2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
         'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
         'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3',
         '4', '5', '6', '7', '8', '9', '!', '?', ' ', ',']
    return W2

def h(entity_A, entity_B, P):
    traces = []
    for rnd in range(1, P["iterations"] + 1):
        # ---- 单步交互：A 的每个值映射到 B 的某个位置 ----
        result = ""
        for value in entity_A:
            index = value % len(entity_B)
            result += entity_B[index]
        traces.append((rnd, result))
        # ---- 迭代：按 P 中定义的规则改变实体 ----
        entity_A = P["mutate_A"](entity_A, rnd)
        entity_B = P["mutate_B"](entity_B, rnd)
    return traces

def pm_P():
    P = {
		"iterations": 5,
		"mutate_A": lambda entity_A, round_num: (
			[v + round_num for v in entity_A]
		),
		"mutate_B": lambda entity_B, round_num: (
			entity_B[round_num % len(entity_B):] +
			entity_B[:round_num % len(entity_B)]
		),
	}
	return P

def g():
    W1 = f1()
    W2 = f2()
    P = pm_P()
    return h(W1, W2, P)
if __name__ == "__main__":
    traces = g()
    for rnd, output in traces:
        print(f"轮次 {rnd}: {output}")
