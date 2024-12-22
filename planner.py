from itertools import permutations
import json

comp_mem_ratio = {'3090': 35.58 / 24.0, '2080ti': 13.45 / 11.0, '2080': 10.07 / 8}
device_map = ['3090', '3090', '2080ti', '2080ti', '2080']
device_to_mem = {'3090':24*1024, '2080ti':11*1024, '2080':8*1024}
L = 24 + 2 + 32

mem = []
mem_list4 = [20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             300, 300,
             70,60,60,60,60,60,60,60,
             60,60,60,60,60,60,60,60,
             60,60,60,60,60,60,60,60,
             60,60,60,60,60,60,60,70]  # 计算每层第五流水级的显存量
mem_list3 = [x * 2 for x in mem_list4]  # 计算每层第四流水级的显存量
mem_list2 = [x * 3 for x in mem_list3]  # 计算每层第三流水级的显存量
mem_list1 = [x * 4 for x in mem_list2]  # 计算每层第二流水级的显存量
mem_list0 = [x * 5 for x in mem_list1]  # 计算每层第一流水级的显存量

mem.append(mem_list0)
mem.append(mem_list1)
mem.append(mem_list2)
mem.append(mem_list3)
mem.append(mem_list4)

stage_start = []
stage_end = []
stage_mem_need = [0, 0, 0, 0, 0]

def DetectedOom(stage, device):
    mem_need = 0.0
    for i in range(stage_start[stage], stage_end[stage] + 1):
        mem_need += mem[stage][i]
    
    if mem_need > device_to_mem[device]:
        print(mem_need, device_to_mem[device])
        return True
    
    stage_mem_need[stage] = mem_need
    return False

def CalculateWeightingMemoryRatio(device_topo):
    T = 0.0
    for i, device in enumerate(device_topo):
        mem_need = stage_mem_need[i]
        T += comp_mem_ratio[device] * (1 - mem_need / device_to_mem[device])
    return T

result_file = "result.json"
results = []
# 遍历每一种设备topo
all_permutations = set(permutations(device_map))
for device_topo in all_permutations:
    # 每种拓扑都找一个方案
    curr = 10000.0
    T = 10000.0
    plan = [0, 0, 0, 0] 
    # 进行流水级的层划分
    for i in range(10, 35):
        for j in range(i + 1, 48):
            for k in range(j + 1 , 54):
                for l in range(k + 1, L):
                    print(i, j, k, l)
                    stage_start.append(0), stage_end.append(i - 1)
                    stage_start.append(i), stage_end.append(j - 1)
                    stage_start.append(j), stage_end.append(k - 1)
                    stage_start.append(k), stage_end.append(l - 1)
                    stage_start.append(l), stage_end.append(L - 1)

                    detected = False
                    for s in range(0, 5):
                        if DetectedOom(s, device_topo[s]):
                            detected = True
                            break

                    if not detected:
                        T = CalculateWeightingMemoryRatio(device_topo)
                        if T < curr:
                            curr = T
                            plan = [i,j,k,l]

                    stage_start = []
                    stage_end = []

    # 记录结果
    result = {"device_topo":device_topo,
              "stage_mem_need":stage_mem_need,
              "plan":plan,
              "T":T}
    results.append(result)


# 记录结果
sorted_results = sorted(results, key=lambda x: x["T"])
with open(result_file, 'w', encoding="utf-8") as f:
    json.dump(sorted_results, f, indent=4, ensure_ascii=False)

print(f"数据已保存到 {result_file} 中")