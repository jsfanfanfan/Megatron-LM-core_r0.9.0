# profile time when different layers on different GPUs
# freeze-LM , mbs=1, tp=1, pp=4, dp=4
# 3090
time_3090_forward_freeze = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                     9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                     11, 11,
                     64,64,64,64,64,64,64,64,64,64,64,140]

time_3090_forward_unfreeze = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                     9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                     11, 11,
                     64,64,64,64,64,64,64,64,64,64,64,140]

time_3090_backward_freeze = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                      9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                      13, 13,
                      77,77,77,77,77,77,77,77,77,77,77,136]

time_3090_backward_unfreeze = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                      9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                      13, 13,
                      77,77,77,77,77,77,77,77,77,77,77,136]
# 2080
time_2080_forward_freeze = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     17, 17,
                     190,190,190,190,190,190,190,190,190,190,190,317]
time_2080_forward_unfreeze = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                     30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                     17, 17,
                     190,190,190,190,190,190,190,190,190,190,190,317]
time_2080_backward_freeze = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     17, 17,
                     175,175,175,175,175,175,175,175,175,175,175,320]
time_2080_backward_unfreeze = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     17, 17,
                     175,175,175,175,175,175,175,175,175,175,175,320]
# 2080ti
time_2080ti_forward_freeze = [i / 1.30 for i in time_2080_forward_freeze]
time_2080ti_forward_unfreeze = [i / 1.30 for i in time_2080_forward_freeze]
time_2080ti_backward_freeze = [i / 1.30 for i in time_2080_backward_freeze]
time_2080ti_backward_unfreeze = [i / 1.30 for i in time_2080_backward_unfreeze]


freeze_layer = []
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
stage_forward_time = [0, 0, 0, 0]
stage_backward_time = [0, 0, 0, 0]
stage_optim_time = [0, 0, 0, 0]

import math
def CalculateTimeVariance(device_topo):
    for i in range(stage):
        for j in range(stage_start[i], stage_end[i] + 1):
            if j in freeze_layer:
                stage_forward_time[i] += time_forward_freeze[device_topo[i]][stage_start[j]]
                stage_backward_time[i] += time_backward_freeze[device_topo[i]][stage_start[j]]
            else:
                stage_forward_time[i] += time_forward_unfreeze[device_topo[i]][stage_start[j]]
                stage_backward_time[i] += time_backward_unfreeze[device_topo[i]][stage_start[j]]
        if i < 2:
            if device_topo[i] == '3090':
                stage_optim_time[i] = 12.0 * (stage_end[i] - stage_start[i] + 1)
            else:
                stage_optim_time[i] = 19.6 * (stage_end[i] - stage_start[i] + 1)
    
    return max(stage_forward_time) + max(stage_backward_time) + max(stage_optim_time)
  

# 遍历每一种设备 topo
all_permutations = set(permutations(device_map))
for device_topo in all_permutations:
    # 每种拓扑都找一个方案
    if device_topo == ('2080', '3090', '3090', '2080ti'):
        print("Yes!")
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
                        if stage_start == [0, 10, 31, 37] and stage_end == [9, 30, 36, 37]:
                            print("---------------", T, curr)
                        if T < curr:
                            curr = T
                            plan = [i, j, k]
                            curr_stage_forward_time = stage_forward_time
                            curr_stage_backward_time = stage_backward_time
                            curr_optim_time = stage_optim_time

                            # 记录结果
                            result = {"device_topo":device_topo,
                                    "forward_time":curr_stage_forward_time,
                                    "backward_time":curr_stage_backward_time,
                                    "optim_time":curr_optim_time,
                                    "plan":plan,
                                    "T":T}
                            results.append(result)

                    stage_start = []
                    stage_end = []

# 记录结果
import json
sorted_results = sorted(results, key=lambda x: x["T"])
with open(result_file, 'w', encoding="utf-8") as f:
    json.dump(sorted_results, f, indent=4, ensure_ascii=False)

print(f"数据已保存到 {result_file} 中")