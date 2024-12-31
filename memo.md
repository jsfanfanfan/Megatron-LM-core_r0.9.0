# Megatron 的初始化和模型划分思路




## LLava 模型结构:

### CLIP-vit 模型 (self.encoder）:
    一个卷积层: self.conv1
    一个 LayerNorm 层: self.ln_pre
    24 层 transformer: self.decoder

### Projector (self.projection):
    mlp 类型: self.linear_fc1 + self.linear_fc2
    affine类型: self.linear_fc1

### Mistral-7B (self.llm):
    一个 embedding 层: self.embedding
    32 层 transformer: self.decoder
    一层 output Linear + LayerNorm: self.output_layer

简化问题为 24 + 2 + 32 = 58 层模型

### Megatron 初始化:
    1. 根据 encoder-pipeline-parallel-size 和 encoder-tensor-pipeline-parallel-size 给 encoder 和 projector 划分 GPU rank,生成并行组
    2. 以 encoder 分配的 GPU 数量作为 offset,根据 pipeline-model-parallel-size 和 tensor-model-parallel-size 在 offset 的基础上给 llm 划分 GPU rank,生成并行组
    3. Megatron v0.9.0 支持 encoder-tensor-pipeline-parallel-size < tensor-pipeline-parallel-size 的情况,比如 encoder TP=2,llm TP=4。Megatorn 通过一层封装对流水线 rank 进行连接,生成流水线并行组。

### Megatron 现在的模型划分:
    encoder 和 projector 捆绑起来,PP 必须小于等于 1,二者有相同的并行策略；
    
    llm 部分直接根据流水级数均分 32 层, 第一个流水级有 embedding, 最后一个流水级有 output_layer。

## 修改思路实现初始化和模型的层划分

    1.抛弃 encoder-pipeline-parallel-size 参数,流水线并行只有一个参数 pipeline-paarllel-size;
    
    2.抛弃 encoder-tensor-parallel-size 参数；encoder,projector,llm 共享 tensor-parallel-size 参数;
    
    3.将模型视为 layer_sum = 24 + 2 + 32 = 58 层的 layer-stack 模型,clip-vit 的 Conv.2D 和 LayerNorm 在第一个流水级上而且不做张量并行,llm 的 embedding 和 output Linear 保持原来的实现。使用一个参数 --split_spec 传递划分列表, len(split_spec) = pipeline-parallel-size, sum(split_spec) = layer_sum
    
    4.限制条件:流水级不能切分在 Projector 内部,即 2 层的 Projector 必须在一个 rank
    （注: 以 5 级流水, 58 层模型为例）
    
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
    
    每个 GPU 获取到的模型块的输入可能是不一样的,解决 forward 的传播问题
    
    初始化问题:抛弃 megatron 的分模块初始化,直接进行初始化。不使用 generator-wrapper,直接进行 tp,pp,dp 并行组的初始化

上周实现了 并行组初始化,模型层的任意划分以及不同模型块的前向传播



目前的问题:流水级之间通信问题（已经有解决思路了,明天晚上之前应该能解决）。

megatron/core/pipeline_parallel/schedules.py 1193 行

p2p 通信接收 tensor 时有张量 size 参数,由于模型异构,这个p2p通信接收张量的 size 有问题。 

## 方案 1:最小化 weighting memory waste ratio

假设流水级数为 S,一个 iteration 内设备 $GPU_i$ 的峰值显存占用表示为 $MaxAllocated_i$,显存容量表示为 $Capacity_i$,所以该设备的 memory waste ratio 表示为:
$$
1 - \frac{MaxAllocated_i}{Capacity_i}
$$

流水线上所有 GPU 的 memoey waste ratio 表示为:
$$
\sum^{S}_{i=1} (1 - \frac{MaxAllocated_i}{Capacity_i})
$$

但是所有 GPU 的显存宝贵程度不一样,算力越大的 GPU 显存越宝贵,所以算存比越高的 GPU 越不应该被浪费,修改一下目标,记 GPU_i 的算力为 $Compute_i$, 最小化:
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

要解决的核心问题:给定 gbs,mbs,tp,L,freeze 怎么计算一个 iteration 中的峰值显存来精准判定 OOM（以 fp32 为例）？ 如何剪枝？

$ Memory(weight)+ Memory(activation) + Memory(optimizer) $

需要 profile 的数据:每一层的参数量计算 Memory(weight),每一层中间激活大小,非冻结层的 Adam optimizer 大小按照参数量 * 3 计算,冻结层算 0, 实际需求内存估计会偏离估计值,GPU 可用显存也不是标称,所以 OOM detected 要进行松弛。以前的层划分策略不考虑流水级位置对于显存的影响,现在需要更准确的考虑。



根据层划分获取每一层的参数量,计算 weight 的大小；
计算中间激活的大小；
计算 optim 的大小




## CLIP_ViT + Mistral-3B

gbs=32, 3090 * 4, 26-4-4-4

no-freeze
    DP=4, TP=1, PP=4, mbs=1 3392 ms/iteration
    8.68G-18.79G-16.30G-18.80G

freeze-LM
    DP=4, TP=1, PP=4, mbs=1 2269 ms/iteration
    8.68G-8.51G-6.50G-5.47G
    DP=4, TP=1, PP=4, mbs=2 2488 ms/iteration
    14.35G-13.04G-9.71G-7.17G

freeze-ViT
    DP=4, TP=1, PP=4, mbs=1 3318ms/iteration
    2.51G-18.79G-16.30G-18.80G


### 异构环境测试

gbs=32, 3090 * 2 + 2080ti + 2080

freeze-LM 
    2080-3090-3090-2080ti
        DP=4, TP=1, PP=4, mbs=1
        9-22-6-1; 2543 ms/iteration
        峰值显存占用:2.92G(8)-15.04G(24)-9.60G(24)-1.93G(11)
        前向传播时间:268-890-780-560
        反向传播时间:315-1040-945-560
        梯度同步时间:183-209-0-0

        10-21-6-1; 2519 ms/iteration
        峰值显存占用:3.24G(8)-14.78G(24)-9.60G(24)-1.93G(11)
        前向传播时间:300-880-780-560
        反向传播时间:350-1030-945-560
        梯度同步时间:197-195-0-0

    2080ti-3090-3090-2080 
        DP=4, TP=1, PP=4, mbs=1
        13-18-6-1; 2513ms/iteration
        峰值显存占用:4.18G(8)-14.00G(24)-9.60G(24)-1.93G(11)
        前向传播时间:370-840-780-640
        反向传播时间:380-990-940-635
        梯度同步时间:143-173-0-0

    3090-3090-2080ti-2080
        DP=4, TP=1, PP=4, mbs=1 
        28-5-3-2; 3227 ms/iteration
        峰值显存占用:13.89G(24)-10.03G(24)-5.07G(11)-3.23G(8)
        前向传播时间:800-1040-990-1040 (均衡各个流水级的时间不可行)
        反向传播时间:910-960-1040-995
        梯度同步时间:311-0-0-0


freeze-ViT
    2080-3090-3090-2080ti
        DP=2, TP=2, PP=4, mbs=1
        26-5-5-2; 3902 ms/iteration
        峰值显存占用:1.26G-11.44G-10.21G-5.36G
        前向传播时间:1540-1080-1005-1160
        反向传播时间:120-1500-1470-1650
        梯度同步时间:60-481-438-238
        optimizer: 8-61-54-46

    3090-3090-2080ti-2080
        DP=2, TP=2, PP=4, mbs=1 
        28-5-3-2; 4638 ms/iteration
        峰值显存占用:6.64G(24)-10.27G(24)-6.20G(11)-5.42G(8)
        前向传播时间:1390-1335-1275-1340
        反向传播时间:800-1800-1960-1940
        梯度同步时间:258-437-275-450
        optimizer: 31-54-52-54

    


no-freeze
    2080-3090-3090-2080ti
        DP=2, TP=2, PP=4, mbs=1
        22-9-5-2; 4060ms/iteration
        峰值显存占用:4.05G(8)-12.40G(24)-10.21G(24)-5.36G(11)
        前向传播时间:1500-1220-1030-1150
        反向传播时间:1270-1600-1480-1610
        梯度同步时间:233-520-435-245
        optimizer: 28-65-54-45

    2080ti-3090-3090-2080
        DP=2, TP=2, PP=4, mbs=1 
        24-8-5-1; 4306ms/iteration
        峰值显存占用:5.03G(11)-11.44G(24)-12.24G(24)-3.33G(11)
        前向传播时间:1460-1270-1010-870
        反向传播时间:1310-1850-1480-1200
        梯度同步时间:125-600-435-280
        optimizer: 25-74-53-33

    3090-3090-2080ti-2080
        DP=2, TP=2, PP=4, mbs=1 
        28-5-3-2; 5195 ms/iteration
        峰值显存占用:9.37G(24)-10.27G(24)-6.20G(11)-5.42G(8)
        前向传播时间:1700-1335-1275-1340
        反向传播时间:1600-1800-1960-1940
        梯度同步时间:378-884-275-450
        optimizer: 31-54-52-54


## CLIP_ViT + Mistral-7B

gbs=32, 3090 * 5, 26-8-8-8-8

no-freeze
    DP=1, TP=4, PP=5, mbs=1 7730 ms/iteration
    3.60G-9.06G-8.18GG-8.18G-8.80G

freeze-LM
    DP=4, TP=1, PP=5, mbs=1 4735 ms/iteration
    5.80G-10.19G-8.28G-6.66G-5.26G

freeze-ViT
    DP=1, TP=4, PP=5, mbs=1 7672ms/iteration
    0.64G-18.79G-16.30G-18.80G


### 异构环境测试

gbs=32, 3090 * 2 + 2080ti * 2 + 2080

no-freeze
    2080-3090-3090-2080ti-2080ti
        DP=1, TP=4, PP=5, mbs=1
        26-11-11-5-5  10153 ms/iteration
        峰值显存占用:5.22G(8)-14.19G(24)-11.22G(24)-5.13G(11)-5.75G
        前向传播时间:3270-4110-3960-3730-3070
        反向传播时间:3140-4680-4680-4200-4040
        optimizer: 20-63-60-44-50

    2080-2080ti-2080ti-3090-3090
        DP=1, TP=4, PP=5, mbs=1
        24-7-6-10-11 10240 ms/iteration
        峰值显存占用:3.18G(8)-6.07G(24)-6.16G(24)-10.21G(11)-11.85G
        前向传播时间:3070-4280-3400-3580-4100
        反向传播时间:2750-4560-4350-4250-4875

        DP=1, TP=4, PP=5, mbs=1
        26-5-6-11-10 10048 ms/iteration
        峰值显存占用:3.47G(8)-5.86G(24)-6.16G(24)-11.22G(11)-10.83G
        前向传播时间:3450-3700-3380-3940-3740
        反向传播时间:2980-4100-4310-4660-4430

freeze-LM 
    2080-3090-3090-2080ti-2080ti
        DP=2, TP=2, PP=5, mbs=1
        24-13-11-5-5 6229 ms/iteration
        峰值显存占用:5.22G(8)-14.19G(24)-11.31G(24)-4.23G(11)-3.43G
        前向传播时间:1590-2220-2200-2170-2330
        反向传播时间:1380-2550-2490-2220-2420
        梯度同步时间:250-31-0-0-0
        optimizer: 30-3-0-0-0
        
        DP=2, TP=2, PP=5, mbs=1
        26-11-11-5-5 6312 ms/iteration
        峰值显存占用:5.82G(8)-13.83G(24)-11.31G(24)-4.23G(11)-3.43G
        前向传播时间:1610-2190-2200-2170-2330
        反向传播时间:1460-2500-2490-2220-2420
        梯度同步时间:305-0-0-0-0
        optimizer: 37-0-0-0-0
    
    2080-2080ti-2080ti-3090-3090
        DP=2, TP=2, PP=5, mbs=1
        20-10-5-12-11 6440 ms/iteration
        峰值显存占用:5.22G(8)-14.19G(24)-11.31G(24)-4.23G(11)-3.43G
        前向传播时间:1590-2220-2200-2170-2330
        反向传播时间:1380-2550-2490-2220-2420
        梯度同步时间:250-31-0-0-0
        optimizer: 30-3-0-0-0


    3090-3090-2080ti-2080ti-2080
        DP=2, TP=2, PP=5, mbs=1
        32-11-5-6-4; 6993 ms/iteration
        峰值显存占用:14.59G(8)-13.60G(24)-5.31G(24)-5.10G(11)-2.88G
        前向传播时间:2490-2840-2100-2420-2500
        反向传播时间:2220-2730-2150-2520-2320
        梯度同步时间:153-0-0-0-0
        optimizer: 20-0-0-0-0


freeze-ViT
    2080-3090-3090-2080ti-2080ti
        DP=1, TP=4, PP=5, mbs=1
        26-11-11-5-5   ms/iteration
        峰值显存占用:5.22G(8)-14.19G(24)-11.22G(24)-5.13G(11)-5.75G
        前向传播时间:3270-4110-3960-3730-3070
        反向传播时间:3140-4680-4680-4200-4040
        optimizer: 20-63-60-44-50

    2080-2080ti-2080ti-3090-3090
        DP=1, TP=4, PP=5, mbs=1
        24-7-6-10-11  ms/iteration
        峰值显存占用:3.18G(8)-6.07G(24)-6.16G(24)-10.21G(11)-11.85G
        前向传播时间:3070-4280-3400-3580-4100
        反向传播时间:2750-4560-4350-4250-4875

        DP=1, TP=4, PP=5, mbs=1
        26-5-6-11-10 9977 ms/iteration
        27-5-6-10-10
        峰值显存占用:3.47G(8)-5.86G(24)-6.16G(24)-11.22G(11)-10.83G
        前向传播时间:3450-3700-3380-3940-3740
        反向传播时间:2980-4100-4310-4660-4430
    



