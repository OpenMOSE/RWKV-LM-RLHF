import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)

ppp = 'myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth'
dim = 4096
GATE_DIM = 512

VER = 6.0
W_LORA_DIM = 128

EMB_DIV = 2
LINEAR_DIV = 2
FFN_DIV = 3.5
GATE_DIV = 2
LN_BIAS_DIV = 2
LN_POWER = 0.5

DecaySize = 128

print('loading...', ppp)
s = torch.load(ppp, map_location="cpu")
keys = list(s.keys())

# for k in keys:
#     print(str(list(s[k].shape)).ljust(15), k.ljust(35))
# exit(0)
'''
v6
[1, 1, 2048]    blocks.0.att.time_maa_x !!
[1, 1, 2048]    blocks.0.att.time_maa_w !!
[1, 1, 2048]    blocks.0.att.time_maa_k
[1, 1, 2048]    blocks.0.att.time_maa_v
[1, 1, 2048]    blocks.0.att.time_maa_r
[1, 1, 2048]    blocks.0.att.time_maa_g
[2048, 160]     blocks.0.att.time_maa_w1 !!
[5, 32, 2048]   blocks.0.att.time_maa_w2 !!
[1, 1, 2048]    blocks.0.att.time_decay !!
[2048, 64]      blocks.0.att.time_decay_w1 !!
[64, 2048]      blocks.0.att.time_decay_w2 !!
[32, 64]        blocks.0.att.time_faaaa
[2048, 2048]    blocks.0.att.receptance.weight
[2048, 2048]    blocks.0.att.key.weight
[2048, 2048]    blocks.0.att.value.weight
[2048, 2048]    blocks.0.att.output.weight
[2048, 2048]    blocks.0.att.gate.weight
[2048]          blocks.0.att.ln_x.weight
[2048]          blocks.0.att.ln_x.bias
[1, 1, 2048]    blocks.0.ffn.time_maa_k
[1, 1, 2048]    blocks.0.ffn.time_maa_r
[7168, 2048]    blocks.0.ffn.key.weight
[2048, 2048]    blocks.0.ffn.receptance.weight
[2048, 7168]    blocks.0.ffn.value.weight

v7-f2a-ka
[1, 1, 768]     blocks.0.att.time_maa_r
[1, 1, 768]     blocks.0.att.time_maa_w
[1, 1, 768]     blocks.0.att.time_maa_k
[1, 1, 768]     blocks.0.att.time_maa_v
[1, 1, 768]     blocks.0.att.time_maa_a
[1, 1, 768]     blocks.0.att.time_maa_g
[1, 1, 768]     blocks.0.att.time_decay
[1, 1, 12, 64]  blocks.0.att.time_faaaa
[1, 1, 768]     blocks.0.att.time_aaaaa
[768, 64]       blocks.0.att.time_decay_w1
[64, 768]       blocks.0.att.time_decay_w2
[768, 64]       blocks.0.att.time_aaa_w1
[64, 768]       blocks.0.att.time_aaa_w2
[1, 1, 768]     blocks.0.att.time_misc_kkk
[1, 1, 768]     blocks.0.att.time_misc_kkb
[768, 128]      blocks.0.att.gate_w1
[128, 768]      blocks.0.att.gate_w2
[1, 1, 768]     blocks.0.att.time_misc_a
[768, 32]       blocks.0.att.mv_w1
[32, 768]       blocks.0.att.mv_w2
[1, 1, 768]     blocks.0.att.time_misc_v
[768, 768]      blocks.0.att.receptance.weight
[768, 768]      blocks.0.att.key.weight
[768, 768]      blocks.0.att.value.weight
[768, 768]      blocks.0.att.output.weight
[768]           blocks.0.att.ln_x.weight
[768]           blocks.0.att.ln_x.bias
[1, 1, 768]     blocks.0.ffn.time_maa_k
[3072, 768]     blocks.0.ffn.key.weight
[768, 3072]     blocks.0.ffn.value.weight
'''

ss = {}
for k in keys:
    print(str(list(s[k].shape)).ljust(15), k.ljust(35), end=' ')

    if 'emb.weight' in k or 'head.weight' in k:
        ss[k] = (s[k] / EMB_DIV).bfloat16()
        print('==> rescale')
    elif 'ffn.receptance' in k or 'time_maa_x' in k or 'time_maa_w1' in k or 'time_maa_w2' in k or 'time_faaaa' in k:
        print('==> REMOVE')
    elif 'att.receptance' in k or 'att.key' in k or 'att.value' in k or 'att.output' in k or 'ffn.key' in k or 'ffn.value' in k:
        if VER == 5.0:
            ss[k] = (s[k] / LINEAR_DIV).bfloat16()
            print('==> rescale')
        if VER > 5.0:
            if 'ffn.key' in k:
                ww = torch.empty(int(dim*3.5), dim).uniform_(-0.001, 0.001)
                ww[:int(dim*3.5), :] = s[k] / FFN_DIV
                ss[k] = ww.bfloat16()
            elif 'ffn.value' in k:
                ww = torch.empty(dim, int(dim*3.5)).uniform_(-0.001, 0.001)
                ww[:, :int(dim*3.5)] = s[k] / FFN_DIV
                ss[k] = ww.bfloat16()
            else:
                ss[k] = (s[k] / LINEAR_DIV).bfloat16()
            print('==> upgrade')
    elif 'att.gate' in k:
        U, S, Vh = torch.linalg.svd(s[k].float())
        n = GATE_DIM
        U_n = U[:, :n]  # Shape [1024, 128]
        S_n = S[:n]     # Shape [128]
        Vh_n = Vh[:n, :] # Shape [128, 1024]
        w1 = U_n * S_n  # Broadcasting S_k to shape [1024, 128]
        w2 = Vh_n       # Shape [128, 1024]
        ss[k.replace('gate.weight', 'gate_w1')] = (w1 / GATE_DIV).bfloat16()
        ss[k.replace('gate.weight', 'gate_w2')] = (w2 / GATE_DIV).bfloat16()
        print('==> w1 w2')
    elif '.ln' in k or 'ln_out' in k:
        if '.bias' in k:
            ss[k] = (s[k] / LN_BIAS_DIV).bfloat16()
            print('==> rescale')
        else:
            ss[k] = (torch.clip(s[k], min=0) ** LN_POWER).bfloat16()
            # print(s[k].float().numpy())
            print('==> normalize')
    elif 'time_mix_' in k:
        if 'ffn.time_mix_r' not in k:
            ss[k.replace('_mix_', '_maa_')] = (1.0 - s[k]).bfloat16()
        print('==> maa')
    elif 'time_maa_' in k and '_w1' not in k and '_w2' not in k:
        if 'ffn.time_maa_r' not in k:
            ss[k] = s[k].bfloat16()
        print('==> maa')
    elif 'time_decay_w1' in k:
        ww = torch.empty(dim, W_LORA_DIM).uniform_(-0.0001, 0.0001)
        ww[:, :DecaySize] = s[k]
        ss[k] = ww.bfloat16()
        print('==> upgrade')
    elif 'time_decay_w2' in k:
        ww = torch.empty(W_LORA_DIM, dim).uniform_(-0.0001, 0.0001)
        ww[:DecaySize, :] = s[k]
        ss[k] = ww.bfloat16()
        print('==> upgrade')
    elif '.time_decay' in k:
        if VER == 5.0:
            ss[k] = s[k].unsqueeze(1).repeat(1, 64).reshape(1,1,-1).bfloat16()
        elif VER == 5.1:
            ss[k] = s[k].reshape(1,1,-1).bfloat16()
        else:
            ss[k] = s[k].bfloat16()
        print('==> upgrade')
    # elif '.time_faaaa' in k:
    #     ss[k] = s[k].reshape(1,1,-1,64).bfloat16()
    #     print('==> upgrade')
    elif '.time_first' in k:
        s[k] = s[k].unsqueeze(1).repeat(1, 64).reshape(1,1,-1,64)
        ss[k.replace('time_first', 'time_faaaa')] = torch.exp(s[k].float()).bfloat16()
        print('==> faaaa')
    else:
        print('?'*100)

print('=' * 80)
# exit(0)

keys = list(ss.keys())
for k in keys:
    print(str(list(ss[k].shape)).ljust(15), k.ljust(35))

torch.save(ss, 'myfolder/converted/x060-upgraded.pth')

exit(0)
