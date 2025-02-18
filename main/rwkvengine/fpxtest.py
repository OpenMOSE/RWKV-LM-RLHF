import torch
from torchao.dtypes.floatx import to_scaled_tc_floatx
from torchao.ops import quant_llm_linear

fp32_weight = torch.randn(1024, 4096).cuda()
ebits, mbits = 3, 2

# pre-process the weight. this will quantize the weight to FP6 and pack it in a special
# layout for tensor cores. refer to paper for more details.
fp6_weight, scales = to_scaled_tc_floatx(fp32_weight, ebits, mbits)



fp16_act = torch.randn(13, 4096).cuda().half()
print(f'fp16_act = {fp16_act.shape}')
outputs = quant_llm_linear(ebits, mbits, fp16_act, fp6_weight, scales)  # shape (1, 1024)
print(outputs)