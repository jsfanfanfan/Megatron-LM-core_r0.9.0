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

