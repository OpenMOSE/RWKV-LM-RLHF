import torch
import torch.nn as nn
from torch.nn import functional as F
import os

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    #@MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))
    


if os.environ["RWKV_TRAIN_TYPE"] == 'infctx':
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, token_amount):
            ctx.save_for_backward(y)
            ctx.token_amount = token_amount
            return loss

        @staticmethod
        def backward(ctx, grad_output): #这个函数会不会影响batch和grad_accu的一致性？感觉上会。梯度累积时，factor变大了。但是只有loss缩放，这里的正则化项反而没有缩放
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            if ctx.token_amount == 0:
                return (grad_output, None, None)
            factor = 1e-4 / ctx.token_amount #这一行类似crossentropy在token上平均。
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            if os.environ.get("WN_FIX_L2WRAP"): #实现batch等价性
                # maxx[maxx<3.]=0. #防止对已经较小的logits值下拉，只对大于阈值的往下拉
                gy.scatter_(-1, ids, maxx * factor * grad_output)
            else:
                gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy, None)
    class MemoryEfficientL2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, attention_mask):
            # 必要な情報のみを保存
            ctx.save_for_backward(y, attention_mask)
            ctx.token_amount = attention_mask.sum().item()
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y, attention_mask = ctx.saved_tensors
            
            with torch.no_grad():
                factor = 1e-4 / ctx.token_amount
                mask = attention_mask.unsqueeze(-1)
                
                gy = torch.zeros_like(y)
                masked_y = y.masked_fill(~mask.bool(), float('-inf'))
                maxx, ids = torch.max(masked_y, -1, keepdim=True)
                
                if os.environ.get("WN_FIX_L2WRAP"):
                    gy.scatter_(-1, ids, maxx * factor * grad_output)
                else:
                    gy.scatter_(-1, ids, maxx * factor)
                
                gy.mul_(mask)  
                
            return (grad_output, gy, None)
    class L2Wrap_infctx(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, factor, currentMask):
            ctx.save_for_backward(y, factor, currentMask)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y, factor, currentMask = ctx.saved_tensors

            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            gy.scatter_(-1, ids, maxx * factor)

            # We ensure the mask is reshaped accordingly, and apply it against gy
            gy = gy * currentMask.reshape(gy.shape[0],gy.shape[1],1) # currentMask[:, None][None, :]
            return (grad_output, gy, None, None)
else:
    class L2Wrap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y):
            ctx.save_for_backward(y)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y = ctx.saved_tensors[0]
            # to encourage the logits to be close to 0
            factor = 1e-4 / (y.shape[0] * y.shape[1])
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            gy.scatter_(-1, ids, maxx * factor)
            return (grad_output, gy)
    class L2Wrap2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, loss, y, y2):
            ctx.save_for_backward(y,y2)
            #ctx.save_for_backward(y2)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            y,y2 = ctx.saved_tensors
            #y2 = ctx.saved_tensors[1]
            # to encourage the logits to be close to 0
            factor = 1e-4 / (y.shape[0] * y.shape[1])
            maxx, ids = torch.max(y, -1, keepdim=True)
            gy = torch.zeros_like(y)
            gy.scatter_(-1, ids, maxx * factor)

            factor2 = 1e-4 / (y2.shape[0] * y2.shape[1])
            maxx2, ids2 = torch.max(y2, -1, keepdim=True)
            gy2 = torch.zeros_like(y2)
            gy2.scatter_(-1, ids2, maxx2 * factor2)
            return (grad_output, gy, gy2)
        


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.001 , epsilon=1e-8):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.epsilon = epsilon

    def forward(self, pred, target): #Approch1
        n_classes = pred.size(-1)
        softplus_pred = F.softplus(pred)
        prob = softplus_pred / (softplus_pred.sum(dim=-1, keepdim=True) + self.epsilon)
        log_prob = torch.log(prob + self.epsilon)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.sum(dim=-1) / n_classes
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss        
        return loss
    
class LabelSmoothingLoss2(nn.Module):
    def __init__(self, smoothing=0.001, epsilon=1e-8):
        super().__init__()
        self.smoothing = smoothing
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        pred: (batch_size, n_classes)
        target: (batch_size)  [正解クラスIndex]
        """
        # 1) softplusを取る
        sp = F.softplus(pred)  # shape: [B, C]

        # 2) 各サンプル毎に合計値を計算 (正規化用)
        sp_sum = sp.sum(dim=-1, keepdim=True) + self.epsilon  # shape: [B, 1]

        # 3) ターゲットクラスのみ gather で取得 (NLL用)
        #    log p_{y_i} = log( sp[y_i] ) - log(sp_sum)
        sp_y = sp.gather(dim=-1, index=target.unsqueeze(-1)) + self.epsilon
        nll_loss = - (torch.log(sp_y) - torch.log(sp_sum)).squeeze(-1)

        # 4) 全クラスの log p_k の和 (スムージング用)
        #    sum_k log p_k = sum_k log(sp_k) - C * log(sp_sum)
        #    ただし sum_k log(sp_k) は sp.log().sum(dim=-1) で計算
        #sp = sp.log()  # メモリ節約のためin-placeでlogをとる (必要ならclone()してもOK)
        sum_log_sp = sp.log().sum(dim=-1)  # shape: [B]
        n_classes = pred.size(-1)
        sum_log_prob = sum_log_sp - n_classes * torch.log(sp_sum.squeeze(-1))
        smooth_loss = - sum_log_prob / n_classes

        # 5) ラベルスムージングの合成
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        return loss