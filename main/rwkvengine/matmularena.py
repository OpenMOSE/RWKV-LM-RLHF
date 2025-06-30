import torch
import torch.nn as nn
from typing import Optional
import types, gc, os, time, re
from typing import List
from torch.nn import functional as F
import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_flush_denormal(True)

current_path = os.path.dirname(os.path.abspath(__file__))

# if torch.version.hip is not None:
#     print('Rocm Backend Detected. currently, disabled cublas matmul')
#     load(
#         name=f"wkv_cuda",
#         sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
#         verbose=True,
#         extra_cuda_cflags=["-fopenmp -ffast-math --gpu-max-threads-per-block=120"],
#         extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
#         is_python_module=False)
#     DISABLE_CUBLAS_GEMM = True
# else:
#     print('CUDA Backend Detected.')
#     load(
#             name=f"wkv_cuda",
#             sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
#             verbose=True,
#             extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
#             extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
#             is_python_module=False)
#     DISABLE_CUBLAS_GEMM = True


MyStatic = torch.jit.script
MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyCompile = torch.compile()


# #@MyStatic
# def cuda_mm8_batch_one(Z:int,B: int, N: int, M: int, x, w, mx, rx, my, ry):
#     # xtype = x.dtype
#     # if x.dtype != torch.float16:
#     #     x = x.to(dtype=torch.float16)
#     # B = B * Z
#     # a_2d = x.view(-1, x.shape[-1])
#     # y = torch.empty((x.shape[0] * x.shape[1],  w.shape[-1]), dtype=torch.float32, device=x.device)
#     # torch.ops.rwkv.mm8_seq(B, N, M, a_2d, w, mx, rx, my, ry, y)
#     # y = y.view(x.shape[0], x.shape[1], w.shape[-1])#.to(dtype=xtype)
#     # return y
#     #xtype = x.dtype
#     if x.dtype != torch.float16:
#         x = x.to(dtype=torch.float16)
#     y = torch.empty((x.shape[0] * x.shape[1],  w.shape[-1]), dtype=torch.float32, device=x.device)
#     torch.ops.rwkv.mm8_seq(B*Z, N, M, x.view(-1, x.shape[-1]), w, mx, rx, my, ry, y)
#     return y.view(x.shape[0], x.shape[1], w.shape[-1])


# #@MyStatic
# def torch_mm8_seq(x, w, mx, rx, my, ry):
#     #xtype = x.dtype
#     if x.dtype != torch.float16:
#         x = x.to(dtype=torch.float16)
#     return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)#.to(dtype=xtype)

# #@MyStatic
# def mm8(x, w, mx, rx, my, ry):
#     if w.device.type == 'cuda' and x.shape[1] == 1 and x.shape[0] <= 32:
#         Z, B, N, M = x.shape[0], x.shape[1], w.shape[0], w.shape[1]
#         return cuda_mm8_batch_one(Z, B, N, M, x, w, mx, rx, my, ry)
#     else:
#         return torch_mm8_seq(x, w, mx, rx, my, ry) #its faster torch mm8 in prefilling

# #@MyStatic
# def custom_matmul(a, b, mx: Optional[torch.Tensor]=None, rx: Optional[torch.Tensor]=None, my: Optional[torch.Tensor]=None, ry: Optional[torch.Tensor]=None) -> torch.Tensor:

#     output_dtype = a.dtype
#     if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
#         assert a.dtype == b.dtype
#         #return matmul_float(a, b, output_dtype=output_dtype)
#         return (a @ b).to(output_dtype)
#     elif b.dtype == torch.uint8:
#         assert mx is not None
#         assert rx is not None
#         assert my is not None
#         assert ry is not None
#         return mm8(a, b, mx, rx, my, ry).to(output_dtype)
#     else:
#         raise ValueError("Unsupported dtype")
    
@MyStatic
def hybrid_matmul(a:torch.Tensor,b:torch.Tensor):
    if b.dtype == torch.float8_e4m3fn:
            #print('fp8')
            #print(f'xr shape = {xr.shape}')
            xg = a
            S0=xg.shape[0]
            S1=xg.shape[1]

            #xg = torch.clamp(xg, min=-448.0, max=448.0)#
            xg = xg.clamp_(min=-448.0, max=448.0)#

            if not xg.is_contiguous():
                xg=xg.contiguous()

            #print(f'xg max = {xg.abs().max()}')
            
            x = torch._scaled_mm(
                xg.view(S0*S1,xg.shape[2]).to(torch.float8_e4m3fn),
                b.t(),
                bias=None,
                out_dtype=a.dtype,
                scale_a=torch.tensor(1.0, device='cuda'),
                scale_b=torch.tensor(1.0, device='cuda'),
                use_fast_accum = True
            )
            #x = x.view(S0, S1, -1) #output_weight.shape[-1]
            return x.view(S0, S1, -1)
    else:
            #x = (x * g).to(dtype=output_weight.dtype) @ output_weight
            return a.to(dtype=b.dtype) @ b