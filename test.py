device_topo = ["2080", "3090", "3090", "2080ti"]

stage_start = [0, 10, 31, 37]
stage_end = [9, 30, 36, 37]
stage = 4
time_forward = {}
time_backward = {}
# 3090
time_3090_forward = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                     9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                     11, 11,
                     64,64,64,64,64,64,64,64,64,64,64,140]

time_3090_backward = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                      9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                      13, 13,
                      77,77,77,77,77,77,77,77,77,77,77,136]
# 2080
time_2080_forward = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     17, 17,
                     190,190,190,190,190,190,190,190,190,190,190,317]
time_2080_backward = [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                     17, 17,
                     175,175,175,175,175,175,175,175,175,175,175,320]
# 2080ti
time_2080ti_forward = [i / 1.2 for i in time_2080_forward]
time_2080ti_backward = [i / 1.2 for i in time_2080_backward]

time_forward['3090'] = time_3090_forward
time_forward['2080ti'] = time_2080ti_forward
time_forward['2080'] = time_2080_forward
time_backward['3090'] = time_3090_backward
time_backward['2080ti'] = time_2080ti_backward
time_backward['2080'] = time_2080_backward

stage_forward_time = [0, 0, 0, 0]
stage_backward_time = [0, 0, 0, 0]
stage_optim_time = [0, 0, 0, 0]

def CalculateTimeVariance(device_topo):
    print(stage_start)
    print(stage_end)
    for i in range(stage):
        print(device_topo[i], time_forward[device_topo[i]], flush=True)
        stage_forward_time[i] = sum(time_forward[device_topo[i]][stage_start[i]:stage_end[i] + 1])
        stage_backward_time[i] = sum(time_backward[device_topo[i]][stage_start[i]:stage_end[i] + 1])
        if i < 2:
            if device_topo[i] == '3090':
                stage_optim_time[i] = 12.0 * (stage_end[i] - stage_start[i] + 1)
            else:
                stage_optim_time[i] = 19.6 * (stage_end[i] - stage_start[i] + 1)
    
    return max(stage_forward_time) + max(stage_backward_time) + max(stage_optim_time)

T = CalculateTimeVariance(device_topo)
print(stage_forward_time)
print(stage_backward_time)
print(stage_optim_time)
print(T)