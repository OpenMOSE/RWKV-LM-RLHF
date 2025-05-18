# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MAMBA configuration"""

import math

from transformers.configuration_utils import PretrainedConfig


class MambaConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MambaModel`]. It is used to instantiate a MAMBA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MAMBA
    [state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the Mamba model.
        hidden_size (`int`, *optional*):
            Dimensionality of the embeddings and hidden states. Default: 2048.
        state_size (`int`, *optional*):
            Shape of the state space latents. Default: 16.
        num_hidden_layers (`int`, *optional*):
            Number of hidden layers in the model. Default: 48.
        layer_norm_epsilon (`float`, *optional*):
            The epsilon to use in the layer normalization layers. Default: 1e-5.
        pad_token_id (`int`, *optional*):
            Padding token id. Default: 0.
        bos_token_id (`int`, *optional*):
            The id of the beginning of sentence token in the vocabulary. Default: 0.
        eos_token_id (`int`, *optional*):
            The id of the end of sentence token in the vocabulary. Default: 0.
        expand (`int`, *optional*):
            Expanding factor used to determine the intermediate size. Default: 2.
        conv_kernel (`int`, *optional*):
            Size of the convolution kernel. Default: 4.
        use_bias (`bool`, *optional*):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block. Default: `False`.
        use_conv_bias (`bool`, *optional*):
            Whether or not to use bias in the convolution layer of the mixer block. Default: `True`.
        hidden_act (`str`, *optional*):
            The non-linear activation function (function or string) in the decoder. Default: `"silu"`.
        initializer_range (`float`, *optional*):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices. Default: 0.1.
        residual_in_fp32 (`bool`, *optional*):
            Whether or not residuals should be in `float32`.
            If set to `False` residuals will keep the same `dtype` as the rest of the model. Default: `True`.
        time_step_rank (`Union[int,str]`, *optional*):
            Rank of the the discretization projection matrix.
            `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`. Default: `"auto"`.
        time_step_scale (`float`, *optional*):
            Scale used used to scale `dt_proj.bias`. Default: 1.0.
        time_step_min (`float`, *optional*):
            Minimum `time_step` used to bound `dt_proj.bias`. Default: 0.001.
        time_step_max (`float`, *optional*):
            Maximum `time_step` used to bound `dt_proj.bias`. Default: 0.1.
        time_step_init_scheme (`float`, *optional*):
            Init scheme used for `dt_proj.weight`. Should be one of `["random","uniform"]`. Default: `"random"`.
        time_step_floor (`float`, *optional*):
            Minimum clamping value of the `dt_proj.bias` layer initialization. Default: 0.0001.
        window_size (`int`, *optional*):
            The window size used for sliding window attention. Default: 2048.
        rescale_prenorm_residual (`bool`, *optional*):
            Whether or not to rescale `out_proj` weights when initializing. Default: `False`.
        use_cache (`bool`, *optional*):
            Whether or not the cache should be used. Default: `True`.


    Example:

    ```python
    >>> from transformers import MambaConfig, MambaModel

    >>> # Initializing a Mamba configuration
    >>> configuration = MambaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MambaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mamba"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        state_size: int = 16,
        num_hidden_layers: int = 48,
        layer_norm_epsilon=1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        initializer_range: str = 0.1,
        residual_in_fp32: bool = False,
        time_step_rank: str = "auto",
        time_step_scale: float = 1.0,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_init_scheme: str = "random",
        time_step_floor: float = 1e-4,
        rescale_prenorm_residual: bool = False,
        use_cache: bool = True,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_norm = fuse_norm

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
