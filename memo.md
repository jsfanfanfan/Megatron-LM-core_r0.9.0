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
    


## weekly

**gbs=32    mbs=2    DP=1    PP=5    TP=4     --freeze-LLM**

1-24 层为 encoder；25-26 层为 projector； 27-58层为 llm

**baseling1  3090 * 5  26-8-8-8-8(Megatron):     7594ms/iteration      35.58*5=177.9 tflops**
**baseline2   26-8-8-8-8(Megatron)，最佳设备排列 2080ti-3090-2080ti-3090-2080     15964ms/iteration** 
**baseline3   在 baseline2 的最佳排列下，按照设备算力划分参数量 28-11-4-11-4     OOM**   

参数量：

pre-encoder 1196032

ViT-transformer：3153664 * 24 层

projector：18350080

pre-llm：33554432

llm-transformer: 54534144 * 32 层

post-llm：33558528



同样的层划分：24-8-8-8-10
不同的 GPU 排列:

2080-2080ti-2080ti-3090-3090: 14000ms/iteration;
2080-3090-3090-2080ti-2080ti: 14418ms/iteration; 
2080-3090-2080ti-3090-2080ti: 16279ms/iteration;
**2080-2080ti-3090-2080ti-3090: 13153ms/iteration;**

2080ti-2080ti-3090-3090-2080: 19198ms/iteration;
2080ti-3090-2080ti-3090-2080: 19213ms/iteration;
2080ti-3090-3090-2080ti-2080: 19136ms/iteration;
2080ti-2080ti-3090-2080-3090: 15188ms/iteration;
2080ti-3090-2080ti-2080-3090: 15165ms/iteration;
2080ti-3090-3090-2080-2080ti: 15489ms/iteration;

**3090-3090-2080ti-2080ti-2080: 19353ms/iteration;**
3090-2080ti-3090-2080ti-2080: 19248ms/iteration;
3090-2080ti-2080ti-3090-2080:  19095ms/iteration;
3090-3090-2080ti-2080-2080ti: 15725ms/iteration;
3090-2080ti-3090-2080-2080ti: 15690ms/iteration;
3090-2080ti-2080ti-2080-3090: 15353ms/iteration;

不同的 GPU 排列可以相差 1.47×



同样的 GPU 排列：2080-2080ti-2080ti-3090-3090
不同的层划分：

10-14-2-14-18(2.5G-2.8G-0.4G-10G-8.4G):   14207ms/iteration
10-14-6-12-16(3.4G-2.0G-4.3G-8.5G-7.5G):   13003ms/iteration
14-10-6-14-14(3.4G-2.0G-4.3G-9.8G-6.6G)：11792ms/iteration
12-12-7-13-14 (2.4G-5.2G-2.9G-9.1G-6.6G):  11760ms/iteration
**12-12-8-13-13(2.4G-6.2G-2.9G-9.1G-6.1G):   11040ms/iteration   **
14-10-8-12-14(3.4G-2.0G-6.2G-8.5G-6.6G):   11860ms/iteration
22-9-9-9-9(5.3G-7.0G-8.6G-6.4G-4.4G):          12998ms/iteration;
26-8-8-8-8(Megatron)      2080 OOM;



目前的最好情况：

**2080ti-2080ti-3090-3090-2080: 26-4-12-12-4  10543ms/iteration  10.07 + 13.45 *2 + 35.58 *2 = 108.13 tflops 算力是 ≈ 60.7%，速度是 ≈ 72% **





**gbs=32    mbs=2    DP=1    PP=5    TP=4     --freeze-ViT**



**baseling1  3090 * 5  26-8-8-8-8(Megatron):   7960ms/iteration   **
**baseline2   26-8-8-8-8(Megatron)，最佳设备排列 2080ti-3090-2080ti-3090-2080i     2080ti_2 OOM** 
**baseline3   按照设备算力划分参数量 28-11-11-4-4     2080 OOM**

**必须进行重新划分：27-7-11-6-7  12945ms/iteration 108.13 tflops 算力是 ≈ 60.7%，速度是 ≈ 61.5%**

冻结对于线性层和 transformer 层反向传播的影响差异很大~~~