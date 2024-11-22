ranks = [0,1,2,3,4]
split_spec = [20,10,10,10,8]

for rank in ranks:
    start_layer = sum(split_spec[:rank]) + 1
    end_layer = sum(split_spec[:rank + 1])
    print(f"rank{rank}:---[{start_layer}, {end_layer}]")