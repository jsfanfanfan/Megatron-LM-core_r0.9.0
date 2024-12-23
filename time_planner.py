# partition: 1-23-2-1-30-1

# profile time when different layers on different GPUs

time_forward = {}
time_backward = {}
# 3090
time_3090_forward = [120, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
             37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
             85, 85,
             570,421,421,421,421,421,421,421,
             421,421,421,421,421,421,421,421,
             421,421,421,421,421,421,421,421,
             421,421,421,421,421,421,421,390]

time_3090_backward = [122, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
             45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
             40, 40,
             430,414,414,414,414,414,414,414,
             414,414,414,414,414,414,414,414,
             414,414,414,414,414,414,414,414,
             414,414,414,414,414,414,414,453]
# 2080ti
time_2080ti_forward = [250, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96,
             96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96,
             95, 95,
             802,600,600,600,600,600,600,600,
             600,600,600,600,600,600,600,600,
             600,600,600,600,600,600,600,600,
             600,600,600,600,600,600,600,850]
time_2080ti_backward = [180, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
             92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
             45, 45,
             532,580,580,580,580,580,580,580,
             580,580,580,580,580,580,580,580,
             580,580,580,580,580,580,580,580,
             580,580,580,580,580,580,580,530]
# 2080
time_2080_forward = [350, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130,
             130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130,
             130, 130,
             1100,900,900,900,900,900,900,900,
             900,900,900,900,900,900,900,900,
             900,900,900,900,900,900,900,900,
             900,900,900,900,900,900,900,1100]
time_2080_backward = [350, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130,
             130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130,
             130, 130,
             1100,900,900,900,900,900,900,900,
             900,900,900,900,900,900,900,900,
             900,900,900,900,900,900,900,900,
             900,900,900,900,900,900,900,1100]

time_forward['3090'] = time_3090_forward
time_forward['2080ti'] = time_2080ti_forward
time_forward['2080'] = time_2080_forward
time_backward['3090'] = time_3090_backward
time_backward['2080ti'] = time_2080ti_backward
time_backward['2080'] = time_2080_backward

device_map = ['3090', '3090', '2080ti', '2080ti', '2080']
from itertools import permutations

result_file = "time_result.json"
results = []
L = 58
stage = 5
stage_start = []
stage_end = []
stage_forward_time = [0, 0, 0, 0, 0]
stage_backward_time = [0, 0, 0, 0, 0]

import math
def CalculateTimeVariance(device_topo):
    for i in range(stage):
        stage_forward_time[i] = sum(time_forward[device_topo[i]][stage_start[i]:stage_end[i] + 1])
        stage_backward_time[i] = sum(time_backward[device_topo[i]][stage_start[i]:stage_end[i] + 1])
    
    f_ = sum(stage_forward_time) / len(stage_forward_time)
    b_ = sum(stage_backward_time) / len(stage_backward_time)
    f_sq, b_sq = 0, 0
    for f in stage_forward_time:
        f_sq += (f - f_) ** 2
    for b in stage_forward_time:
        b_sq += (b - b_) ** 2

    return f_ * math.sqrt(f_sq / 4.0) + b_ * math.sqrt(b_sq / 4.0)
    

# 遍历每一种设备topo
all_permutations = set(permutations(device_map))
for device_topo in all_permutations:
    # 每种拓扑都找一个方案
    curr = 1e20
    T = 10000.0
    plan = [0, 0, 0, 0]
    curr_stage_forward_time = [0, 0, 0, 0, 0]
    curr_stage_backward_time = [0, 0, 0, 0, 0] 
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
                    
                    """ 先不进行 OOM 检查
                    detected = False
                    for s in range(0, 5):
                        if DetectedOom(s, device_topo[s]):
                            detected = True
                            break
                    """
                    detected = False
                    if not detected:
                        T = CalculateTimeVariance(device_topo)
                        if T < curr:
                            curr = T
                            plan = [i, j, k, l]
                            curr_stage_forward_time = stage_forward_time
                            curr_stage_backward_time = stage_backward_time

                    stage_start = []
                    stage_end = []

    # 记录结果
    result = {"device_topo":device_topo,
              "forward_time":stage_forward_time,
              "backward_time":stage_backward_time,
              "plan":plan,
              "T":T}
    results.append(result)


# 记录结果
import json
sorted_results = sorted(results, key=lambda x: x["T"])
with open(result_file, 'w', encoding="utf-8") as f:
    json.dump(sorted_results, f, indent=4, ensure_ascii=False)

print(f"数据已保存到 {result_file} 中")