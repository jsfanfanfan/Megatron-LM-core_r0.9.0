# Megatron 的初始化和模型划分思路




## LLava 模型结构：

### CLIP-vit 模型 (self.encoder）：
    一个卷积层: self.conv1
    一个 LayerNorm 层: self.ln_pre
    24 层 transformer: self.decoder

### Projector (self.projection)：
    mlp 类型: self.linear_fc1 + self.linear_fc2
    affine类型: self.linear_fc1

### Mistral-7B (self.llm)：
    一个 embedding 层: self.embedding
    32 层 transformer: self.decoder
    一层 output Linear + LayerNorm: self.output_layer

简化问题为 24 + 2 + 32 = 58 层模型

### Megatron 初始化：
    1. 根据 encoder-pipeline-parallel-size 和 encoder-tensor-pipeline-parallel-size 给 encoder 和 projector 划分 GPU rank，生成并行组
    2. 以 encoder 分配的 GPU 数量作为 offset，根据 pipeline-model-parallel-size 和 tensor-model-parallel-size 在 offset 的基础上给 llm 划分 GPU rank，生成并行组
    3. Megatron v0.9.0 支持 encoder-tensor-pipeline-parallel-size < tensor-pipeline-parallel-size 的情况，比如 encoder TP=2，llm TP=4。Megatorn 通过一层封装对流水线 rank 进行连接，生成流水线并行组。

### Megatron 现在的模型划分：
    encoder 和 projector 捆绑起来，PP 必须小于等于 1，二者有相同的并行策略；
    
    llm 部分直接根据流水级数均分 32 层， 第一个流水级有 embedding， 最后一个流水级有 output_layer。

## 修改思路实现初始化和模型的层划分

    1.抛弃 encoder-pipeline-parallel-size 参数，流水线并行只有一个参数 pipeline-paarllel-size;
    
    2.抛弃 encoder-tensor-parallel-size 参数；encoder，projector，llm 共享 tensor-parallel-size 参数;
    
    3.将模型视为 layer_sum = 24 + 2 + 32 = 58 层的 layer-stack 模型，clip-vit 的 Conv.2D 和 LayerNorm 在第一个流水级上而且不做张量并行，llm 的 embedding 和 output Linear 保持原来的实现。使用一个参数 --split_spec 传递划分列表, len(split_spec) = pipeline-parallel-size, sum(split_spec) = layer_sum
    
    4.限制条件：流水级不能切分在 Projector 内部，即 2 层的 Projector 必须在一个 rank
    （注： 以 5 级流水， 58 层模型为例）
    
    model = model_provider_func(
                add_encoder=add_encoder,
                encoder_pre_process=encoder_pre_process,
                add_projector=add_projector,
                add_decoder=add_decoder
                pre_process=llm_pre_process,
                post_process=llm_post_process)
    
        流水级只有 CLIP-vit 的部分 transformer 层(带不带头);
        流水级有 CLIP-vit 部分 transformer 层（带不带头） Projector 的 Linear 层;
        流水级有 CLIP-vit 部分 transformer 层（带不带头）和所有 Projector 的 Linear 层和部分 llm 的transformer 层（pre_process, 或许 post_precess）;
        流水级只有全部 Projector 层;
        流水级有全部 Projector Linear 层 和部分 llm 的 transformer 层(pre_process, 或许 post_process);
        流水级有部分 llm 的 transformer 层（有头？有尾？二者都有？）;
    
    用 start_layer 和 end_layer 判断各个标识位；
    层数问题在 config 中添加参数一起传过去；
    
    每个 GPU 获取到的模型块的输入可能是不一样的，解决 forward 的传播问题
    
    初始化问题：抛弃 megatron 的分模块初始化，直接进行初始化。不使用 generator-wrapper，直接进行 tp，pp，dp 并行组的初始化

上周实现了 并行组初始化，模型层的任意划分以及不同模型块的前向传播



目前的问题：流水级之间通信问题（已经有解决思路了，明天晚上之前应该能解决）。

megatron/core/pipeline_parallel/schedules.py 1193 行

p2p 通信接收 tensor 时有张量 size 参数，由于模型异构，这个p2p通信接收张量的 size 有问题。 

## 方案 1：最小化 weighting memory waste ratio

假设流水级数为 S，一个 iteration 内设备 $GPU_i$ 的峰值显存占用表示为 $MaxAllocated_i$，显存容量表示为 $Capacity_i$，所以该设备的 memory waste ratio 表示为：
$$
1 - \frac{MaxAllocated_i}{Capacity_i}
$$

流水线上所有 GPU 的 memoey waste ratio 表示为：
$$
\sum^{S}_{i=1} (1 - \frac{MaxAllocated_i}{Capacity_i})
$$

但是所有 GPU 的显存宝贵程度不一样，算力越大的 GPU 显存越宝贵，所以算存比越高的 GPU 越不应该被浪费，修改一下目标，记 GPU_i 的算力为 $Compute_i$, 最小化:
$$
T = \sum^{S}_{i=1} \frac{Compute_i}{Capacity_i} (1 - \frac{MaxAllocated_i}{Capacity_i})
$$

### Algorithm:

initial($P_{best}$, $L_{best}$, $T^*=+∞$) \
for P in Cut{ GPU_Permutation(GPUs) }: \
&nbsp; &nbsp; &nbsp; &nbsp; for L in Cut{ Partitions(Module Layers) }: \
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Calculate(MaxAllocated) \
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if OOM detected: \
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Continue \
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; else: \
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Calculate(T) \
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;if T* < T: \
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; $T^* = T; L_{best}=L; P_{best}=P$ \
return $P_{best}, L_{best}$

要解决的核心问题：给定 gbs，mbs，tp，L，freeze 怎么计算一个 iteration 中的峰值显存来精准判定 OOM（以 fp32 为例）？ 如何剪枝？

$ Memory(weight)+ Memory(activation) + Memory(optimizer) $

需要 profile 的数据：每一层的参数量计算 Memory(weight)，每一层中间激活大小，非冻结层的 Adam optimizer 大小按照参数量 * 3 计算，冻结层算 0, 实际需求内存估计会偏离估计值，GPU 可用显存也不是标称，所以 OOM detected 要进行松弛。以前的层划分策略不考虑流水级位置对于显存的影响，现在需要更准确的考虑。



根据层划分获取每一层的参数量，计算 weight 的大小；
计算中间激活的大小；
计算 optim 的大小




## 不使用 Mistral-7B，换成 Mistral-3B，LLM 变成 12 层

gbs=16， 3090 * 4，26-4-4-4 

no-freeze mbs=1，tp=2，pp=4，dp=2 峰值显存占用：5.01G-9.48G-8.23G-9.48G     2526ms/iteration \
freeze-ViT mbs=1，tp=2，pp=4，dp=2 峰值显存占用：1.28G-9.48G-8.23G-9.48G    2405ms/iteration \
freeze-LM mbs=1，tp=1，pp=4，dp=4 峰值显存占用：8.7G-8.6G-6.62G-5.59G   1652ms/iteration

1.去掉 tp 后，对于一个 transformer 层，反向传播时间不是前向传播时间的严格 2 倍，实际观察应该是 1.6-1.9 倍之间

2.冻结对于 transformer 层的反向传播时间的影响大致降为原来的一半，实际测试中在 0.5-0.6 倍之间，这导致冻结后的层反向传播的时间略低于前向传播的时间

3.如果冻结最前面的模块（encoder）, 反向传播构建计算图时会直接删除 encoder，encoder 不参与反向传播



### 异构环境测试

gbs=16，3090 * 2 + 2080ti + 2080

freeze-LM 时, mbs=1, tp=1, pp=4, dp=4: 
    2080-3090-3090-2080ti
        9-22-6-1; 1542 ms/iteration
        峰值显存占用：2.92G(8)-15.03G(24)-9.60G(24)-1.93G(11)
        前向传播时间：134-447-390-264
        反向传播时间：160-517-467-275
        梯度同步时间：177-207-0-0

        10-21-6-1; 1531 ms/iteration
        峰值显存占用：3.23G(8)-15.28G(24)-9.60G(24)-1.93G(11)
        前向传播时间：148-440-390-264
        反向传播时间：174-510-467-275
        梯度同步时间：195-193-0-0

        11-20-6-1; 1549 ms/iteration
        峰值显存占用：3.54G(8)-15.53G(24)-9.60G(24)-1.93G(11)
        前向传播时间：158-432-390-264
        反向传播时间：192-504-467-275
        梯度同步时间：217-181-0-0

    2080-3090-2080ti-3090
        10-21-2-5; 1621 ms/iteration
        峰值显存占用：3.23G(8)-15.28G(24)-3.40G(11)-6.66G(24)
        前向传播时间：148-445-321-395
        反向传播时间：175-516-348-447
        梯度同步时间：200-192-0-0

        10-21-1-6; 1652 ms/iteration
        峰值显存占用：3.23G(8)-15.28G(24)-3.40G(11)-6.66G(24)
        前向传播时间：148-445-161-445
        反向传播时间：175-510-163-524
        梯度同步时间：200-192-0-0


    2080ti-3090-3090-2080 
        9-22-6-1; 1591 ms/iteration
        峰值显存占用：2.92G(8)-15.03G(24)-9.60G(24)-1.93G(11)
        前向传播时间：132-449-390-317
        反向传播时间：160-518-467-320
        梯度同步时间：188-205-0-0

        10-21-6-1; 1598ms/iteration
        峰值显存占用：3.23G(8)-15.28G(24)-9.60G(24)-1.93G(11)
        前向传播时间：148-442-390-264
        反向传播时间：150-510-467-314
        梯度同步时间：203-191-0-0

    3090-3090-2080ti-2080 
        28-5-3-2; 2120 ms/iteration
        峰值显存占用：13.89G(24)-10.03G(24)-5.07G(11)-3.23G(8)
        前向传播时间：426-515-482-507
        反向传播时间：454-472-503-494
        梯度同步时间：290-0-0-0



freeze-ViT 时, mbs=1, tp=2, pp=4, dp=2:
    2080-3090-3090-2080ti
        26-5-5-2; 2401 ms/iteration
        峰值显存占用：1.26G-11.44G-10.21G-5.36G
        前向传播时间：770-535-503-533
        反向传播时间：60-757-730-804
        梯度同步时间：60-484-446-280
        optimizer: 8-61-54-46


no-freeze 时, mbs=1, tp=2, pp=4, dp=2:
    2080-3090-3090-2080ti
        24-7-5-2; 2582ms/iteration
        峰值显存占用：4.42G-11.76G-10.21G-5.36G
        前向传播时间：770-542-500-570
        反向传播时间：695-775-730-825
        梯度同步时间：250-510-435-280
        optimizer: 31-64-54-46


    2080ti-3090-3090-2080 
        24-7-6-1; 2597ms/iteration
        峰值显存占用：5.03G-11.44G-12.24G-3.33G
        前向传播时间：720-546-595-430
        反向传播时间：630-775-875-600
        梯度同步时间：150-510-515-285
        optimizer: 25-64-64-33



