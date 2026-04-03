import random

def apply_bid(n, b):
    if not (0 <= b <= n):
        raise ValueError("b must be between 0 and n")
    if random.random() < b / n:
        return b
    else:
        return n - b

def is_terminal(n):
    return n == 1

def game(n, k):
    turns_taken = 0
    while True:
        if is_terminal(n):
            return turns_taken
        turns_taken += 1
        b = n // k
        b = max(1, min(b, n - 1))   # ensures progress (no b=0, no b=n)
        n = apply_bid(n, b)

expected_turns = []
n = 20
trials = 100

for k in range(2, n // 2 + 1):      # start at 2 (skip k=1)
    total = 0
    for _ in range(trials):
        total += game(n, k)
    expected_turns.append([k, total / trials])

print(expected_turns)
