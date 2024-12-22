from itertools import permutations

x = [1, 1, 2, 2, 3]

all_ = set(permutations(x))

for i in all_:
    print(list(i))