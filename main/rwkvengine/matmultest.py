import triton
import triton.language as tl
import numpy as np
import torch

device = 'cuda'

f32_type = torch.float32
bf16_type = torch.bfloat16
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2
 

output, output_amax = torch._scaled_mm(
        torch.randn(16,16, device=device).to(e4m3_type),
        torch.randn(16,16, device=device).to(e4m3_type).t(),
        bias=None,#torch.randn(16, device=device).to(bf16_type),
        out_dtype=e4m3_type,
        scale_a=torch.tensor(1.0, device=device),
        scale_b=torch.tensor(1.0, device=device)
    )

print(output)