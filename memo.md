# Megatron 的初始化和模型划分思路

## LLava 模型结构：

### CLIP-vit 模型(self.encoder）：
    一个卷积层: self.conv1
    一个 LayerNorm 层: self.ln_pre
    24 层 transformer: self.decoder

### Projector(self.encoder)：
    mlp 类型: self.linear_fc1 + self.linear_fc2
    affine类型: self.linear_fc1

### Mistral-7B(self.llm)：
    一个 embedding: self.embedding
    32 层 transformer: self.decoder
    一层输出 Linear + LayerNorm: self.output_layer

### Megatron 初始化：
    1. 根据 encoder-pipeline-parallel-size 和 encoder-tensor-pipeline-parallel-size 给 encoder 和 projector 划分 GPU rank，生成并行组
    2. 以 encoder 分配的 GPU 数量作为 offset，根据 pipeline-parallel-size 和 tensor-pipeline-parallel-size 在 offset 的基础上给 llm 划分 GPU rank，生成并行组
    3. Megatron v0.9.0 支持 encoder-tensor-pipeline-parallel-size < tensor-pipeline-parallel-size 的情况，比如 encoder TP=2，llm TP=4。Megatorn 对流水线 rank 进行连接，生成整条流水线并行组。

### Megatron 模型划分：
    encoder 和 projector 捆绑起来，PP 必须小于等于 1，二者有相同的并行策略。
    llm 部分直接根据流水级数均分 32 层， 第一个流水级有 embedding， 最后一个流水级有 output_layer。

# 我的修改思路 —— 如何实现整个模型的层划分

    1.抛弃 encoder-pipeline-parallel-size 参数，流水线并行只有一个参数 pipeline-paarllel-size;

    2.encoder，projector，llm 共享 tensor-parallel-size 参数;

    3.将模型视为 layer_sum = 24 + (2 or 1) + 32 = 57 or 58 层的整体模型，clip-vit 的卷积层和 LayerNorm 在第一个流水级上而且不做张量并行，llm 的 embedding 和 Linear 保持原来的实现。使用一个参数 --split_spec 传递划分列表, len(split_spec) = pipeline-parallel-size, sum(split_spec) = layer_sum

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
    层数问题使用在config中添加参数一起传过去；

    每个 GPU 获取到的模型块的输入可能是不一样的，解决 forward 的传播问题

    初始化问题：抛弃 megatron 的分模块初始化，直接进行初始化。不使用 generator-wrapper，直接进行 tp，pp，dp 并行组的初始化
    




