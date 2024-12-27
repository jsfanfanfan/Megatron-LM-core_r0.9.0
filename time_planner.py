# profile time when different layers on different GPUs
# freeze-LM , mbs=1, tp=1, pp=4, dp=4
# 3090
time_3090_forward_freeze = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     11, 11,
                     63,63,63,63,63,63,63,63,63,63,63,140]

time_3090_backward_freeze = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     13, 13,
                     77,77,77,77,77,77,77,77,77,77,77,136]

time_3090_forward_unfreeze = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     11, 11,
                     63,63,63,63,63,63,63,63,63,63,63,140]

time_3090_backward_unfreeze = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                     9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                     13, 13,
                     130,130,130,130,130,130,130,130,130,130,130,230]
# 2080ti
time_2080ti_forward_freeze = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                     12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                     21, 21,
                     175,165,165,165,165,165,165,165,165,165,165,260]

time_2080ti_backward_freeze = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     24, 24,
                     170,167,167,167,167,167,167,167,167,167,167,275]

time_2080ti_forward_unfreeze = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                     12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                     21, 21,
                     175,165,165,165,165,165,165,165,165,165,165,260]

time_2080ti_backward_unfreeze = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     24, 24,
                     320,303,303,303,303,303,303,303,303,303,303,484]
# 2080
time_2080_forward_freeze = [i * 1.15 for i in time_2080ti_forward_freeze]
time_2080_forward_unfreeze = [i * 1.15 for i in time_2080ti_forward_freeze]
time_2080_backward_freeze = [i * 1.15 for i in time_2080ti_backward_freeze]
time_2080_backward_unfreeze = [i * 1.15 for i in time_2080ti_backward_unfreeze]

time_forward_freeze = {}
time_forward_unfreeze = {}
time_backward_freeze = {}
time_backward_unfreeze = {}

time_forward_freeze['3090'] = time_3090_forward_freeze
time_backward_freeze['3090'] = time_3090_backward_freeze
time_forward_unfreeze['3090'] = time_3090_forward_unfreeze
time_backward_unfreeze['3090'] = time_3090_backward_unfreeze

time_forward_freeze['2080ti'] = time_2080ti_forward_freeze
time_backward_freeze['2080ti'] = time_2080ti_backward_freeze
time_forward_unfreeze['2080ti'] = time_2080ti_forward_unfreeze
time_backward_unfreeze['2080ti'] = time_2080ti_backward_unfreeze

time_forward_freeze['2080'] = time_2080_forward_freeze
time_backward_freeze['2080'] = time_2080_backward_freeze
time_forward_unfreeze['2080'] = time_2080_forward_unfreeze
time_backward_unfreeze['2080'] = time_2080_backward_unfreeze

device_map = ['3090', '3090', '2080ti', '2080']
from itertools import permutations

result_file = "time_result.json"
results = []
L = 38
stage = 4
stage_start = []
stage_end = []

# 指定冻结层
freeze_ViT = False
freeze_LM = True
freeze_layer = []
if freeze_LM:
    freeze_layer = [i for i in range(26, 38)]
elif freeze_ViT:
    freeze_layer = [i for i in range(0, 24)]
else:
    freeze_layer = []

stage_forward_time = [0, 0, 0, 0]
stage_backward_time = [0, 0, 0, 0]
stage_optim_time = [0, 0, 0, 0]

import math
def CalculateTimeVariance(device_topo):
    for i in range(stage):
        for j in range(stage_start[i], stage_end[i] + 1):
            if j in freeze_layer:
                stage_forward_time[i] += time_forward_freeze[device_topo[i]][j]
                stage_backward_time[i] += time_backward_freeze[device_topo[i]][j]
            else:
                stage_forward_time[i] += time_forward_unfreeze[device_topo[i]][j]
                stage_backward_time[i] += time_backward_unfreeze[device_topo[i]][j]
        if i < 2:
            if device_topo[i] == '3090':
                stage_optim_time[i] = 12.0 * (stage_end[i] - stage_start[i] + 1)
            else:
                stage_optim_time[i] = 19.6 * (stage_end[i] - stage_start[i] + 1)

    f_ = sum(stage_forward_time) / len(stage_forward_time)
    b_ = sum(stage_backward_time) / len(stage_backward_time)
    f_pow, b_pow = 0.0, 0.0
    for i in stage_forward_time:
        f_pow += pow(i - f_, 2)
    for j in stage_backward_time:
        b_pow += pow(j - b_, 2)
    
    # return max(stage_forward_time) + max(stage_backward_time) + max(stage_optim_time)
    return pow(f_pow / 4.0, 0.5) + pow(b_pow / 4.0, 0.5) + max(stage_optim_time)
  

# 遍历每一种设备 topo
all_permutations = set(permutations(device_map))
for device_topo in all_permutations:
    # 每种拓扑都找一个方案
    curr = 1e20
    T = 1e10
    plan = [0, 0, 0]
    curr_stage_forward_time = [0, 0, 0, 0]
    curr_stage_backward_time = [0, 0, 0, 0] 
    # 进行流水级的层划分
    for i in range(6, 30):
        for j in range(i + 1, L - 1):
            for k in range(j + 1 , L):
                stage_start.append(0), stage_end.append(i - 1)
                stage_start.append(i), stage_end.append(j - 1)
                stage_start.append(j), stage_end.append(k - 1)
                stage_start.append(k), stage_end.append(L - 1)
                    
                detected = False
                if not detected:
                    T = CalculateTimeVariance(device_topo)
                    if T < curr:
                        curr = T
                        plan = [i, j, k]
                        curr_stage_forward_time = stage_forward_time
                        curr_stage_backward_time = stage_backward_time
                        curr_optim_time = stage_optim_time

                stage_start = []
                stage_end = []
                stage_forward_time = [0, 0, 0, 0]
                stage_backward_time = [0, 0, 0, 0]
                stage_optim_time = [0, 0, 0, 0]

    # 记录结果
    result = {"device_topo":device_topo,
            "forward_time":curr_stage_forward_time,
            "backward_time":curr_stage_backward_time,
            "optim_time":curr_optim_time,
            "plan":plan,
            "T":curr}
    results.append(result)

# 记录结果
import json
sorted_results = sorted(results, key=lambda x: x["T"])
with open(result_file, 'w', encoding="utf-8") as f:
    json.dump(sorted_results, f, indent=4, ensure_ascii=False)

print(f"数据已保存到 {result_file} 中")