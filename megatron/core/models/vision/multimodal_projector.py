# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from megatron.core import tensor_parallel
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor

import torch


class MultimodalProjector(MegatronModule):
    """
    MultimodalProjector will take the encoded input with input_size hidden state and project
    it into the hidden size of the language model for multimodal training. When projector is
    type affine linear_fc1 from submodules is used.

    Args:
        transformer_config (TransformerConfig): Transformer config
        submodules (MLPSubmodules): Specifies MLP submodules for mlp type projector
        projector_type (str): Projector type
        input_size (int): Input size from feature encoder
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        projector_type: str,
        input_size: int,
        pre_process: bool = True,
        post_process: bool = True,
        encoder_pre_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        add_projector: bool = True,
        projector_finished: bool = True,
    ):

        super().__init__(config=config)
        self.projector_type = projector_type
        self.pre_process = pre_process
        self.post_process = post_process
        self.encoder_pre_process = encoder_pre_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.add_projector = add_projector
        self.projector_finished = projector_finished
        assert submodules is not None, "MLPSubmodules must be provided"

        if self.projector_type == "mlp": # 2 个 Linear 层
            self.encoder = MLP(config=config, submodules=submodules, input_size=input_size,
                               add_encoder=add_encoder, add_projector=add_projector, add_decoder=add_decoder)
        elif self.projector_type == "affine":
            self.encoder = build_module( # 1 个 Linear 层
                submodules.linear_fc1,
                input_size,
                config.hidden_size,
                config=config,
                init_method=config.init_method,
                gather_output=True,
                bias=config.add_bias_linear,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name=None,
            )
        else:
            raise Exception(f"Unsupported multimodal projection type {self.projector_type}")
        
    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """给 clip-vit 的 self.decoder(transformer层, 一个 transformer_block 类) 设置输入

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.encoder.set_input_tensor(input_tensor)

    def forward(self, hidden_states):
        """接收 encoder 最后一层 transformer 的 hidden states, 生成 encoder_output"""
        # print(f"mlp forward begin hidden states:{hidden_states.size()}")
        encoder_output, encoder_output_bias = self.encoder(hidden_states)
        # print(f"mlp forward end hidden states:{hidden_states.size()}")
        if encoder_output_bias is not None:
            encoder_output = encoder_output + encoder_output_bias

        # the encoder produces "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        encoder_output = make_viewless_tensor(
            inp=encoder_output, requires_grad=True, keep_graph=True
        )

        return encoder_output
