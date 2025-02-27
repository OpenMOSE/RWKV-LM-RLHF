import os, math, time, datetime, subprocess
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from .model import LAYER_CONFIG
from .layerprofiler import v7_additional_parameters
import gc

def my_save(args, trainer, dd, ff):
    if '14b-run1' in ff:
        fn = ff.split('/')[-1]
        fff = '/dev/shm/' + fn
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif ('world/14b' in ff) or ('world/7b' in ff):
        aa = ff.split('/')[1]
        fn = ff.split('/')[-1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True)
    else:
        if 'deepspeed_stage_3' in args.strategy:
            trainer.save_checkpoint(ff, weights_only=True)
        else:
            torch.save(dd, ff)

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.firsttime = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        if self.firsttime:
            self.firsttime = False
            print(f"Current device: {pl_module.device}")
        	# pl_moduleに含まれるすべてのサブモジュールを指定されたデバイスに移動
            pl_module.to(pl_module.device)
		

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        #print(2)
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))

        #print(4)
        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)
        #print(5)
        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay
        #print(6)


        if args.lr_advanced:
            for param_group in trainer.optimizers[0].param_groups:
                #pname = param_group["pname"]
                param_lr_init = param_group["lr_init"]
                param_lr_final = param_group["lr_final"]
                
                if trainer.global_step < w_step:
                    param_group["lr"] = param_lr_init * (0.2 + 0.8 * trainer.global_step / w_step)
                else:
                    param_group["lr"] = param_lr_init * math.exp(math.log(param_lr_final / param_lr_init) * pow(progress, 1))

                #print(f'Setting {pname} LR_init {param_lr_init} LR_final {param_lr_final} LR_Now {param_group["lr"]}')
        else:
            for param_group in trainer.optimizers[0].param_groups:
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_now
                if args.layerwise_lr > 0:
                    param_group["lr"] = lr * param_group["my_lr_scale"]
                else:
                    param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_step} {lr}")
        #print(7)
        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    #print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args

        if args.distillation or args.sft:
            token_per_step = trainer.realproceedtokens * args.real_bsz
        else:
            token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]*trainer.accumulate_grad_batches
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0: #"wd": trainer.my_wd
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                if args.distillation:
                    try:
                        lll |= {"smooth_loss": trainer.smooth_loss, "kl_loss": trainer.kl_loss, "active_ctx":trainer.realproceedtokens}
                    except: pass
                if args.zerocot:
                    #
                    try:
                        lll |= {"advantage": trainer.advantage,
                                "advantage_clipped": trainer.advantage_clipped,
                                "actor_ppl":trainer.actor_ppl,
                                "base_ppl":trainer.base_ppl,
                                "actor_nll":trainer.actor_nll,
                                "base_nll":trainer.base_nll,
                                "entropy":trainer.entropy,
                                "loss_rl":trainer.loss_rl,
                                "loss_entropy":trainer.loss_entropy                                
                                }
                    except: pass
                if args.grpo:
                    #
                    try:
                        lll |= {"rewards_mean": trainer.rewards_mean,
                                "rewards_std": trainer.rewards_std,
                                "kl_value":trainer.kl_value,
                                "loss_reinforce":trainer.loss_reinforce                                                           
                                }
                    except: pass
                if args.sft:
                    try:
                        lll |= {"smooth_loss": trainer.smooth_loss, "active_ctx":trainer.realproceedtokens}
                        if args.moe:
                            #moe_router_loss
                            lll |= {"moe_router_loss": trainer.moe_router_loss}
                    except: pass
                if args.dpo or args.simpo or args.wpo:
                    try:
                        lll |= {"pref_percentage": trainer.pref_match_percentage, "loss_1": trainer.loss_1_general_or_sft, "loss_2_dpo": trainer.loss_2_dpo}
                    except: pass
                if args.orpo:
                    try:
                        lll |= {"pref_percentage": trainer.pref_match_percentage, "loss_1_token_chosen": trainer.loss_1_general_or_sft, "loss_2_odds_ratio": trainer.loss_2_orpo}
                    except: pass
                trainer.my_wandb.log(lll, step=int(real_step))
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy): # save pth
            if args.magic_prime > 0:
                expand_factor = 2 if args.my_qa_mask > 0 else 1
                if int(real_step) == int(args.magic_prime * expand_factor // args.real_bsz) - 1 + int(args.my_random_steps):
                    to_save_dict = pl_module.state_dict()
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        assert "DPODataset" in str(dataset) or 'HDF5TopKTensorDataset' in str(dataset) or 'RLHFDataset' in str(dataset) or 'JSONLOnDemandOffsetDataset' in str(dataset)
        if args.dpo or args.orpo:
            dataset.global_rank = trainer.global_rank
            dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset.world_size = trainer.world_size
        else:
            dataset.global_rank = trainer.global_rank
            dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        gc.collect()
        torch.cuda.empty_cache()
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                to_save_dict = pl_module.state_dict()

                lora_dict = {}
                state_dict = {}

                param_dict = {n: p for n, p in pl_module.named_parameters()}

                for name, state in to_save_dict.items():
                    #print(f'{name} {param_dict[name].requires_grad}')
                    try:
                        if param_dict[name].requires_grad:
                            if LAYER_CONFIG['emb']['mode']=='full' and 'emb' in name:
                                lora_dict[name] = state
                            if LAYER_CONFIG['head']['mode']=='full' and 'head' in name:
                                lora_dict[name] = state
                            for i in range(args.n_layer):
                                text = f'blocks.{str(i)}'
                                if LAYER_CONFIG[f'{str(i)}']['mode']=='full' and text in name:
                                    lora_dict[name] = state
                                    break
                            if '.time_state' in name and args.state_output_mode == 0:
                                    lora_dict[name] = state
                            elif '.time_state' in name:
                                    lora_dict[name] = state
                                    state_dict[name] = state
                            elif ('.bone' in name or '.lora_' in name or '.time' in name or 'ln' in name or 'router' in name):
                                lora_dict[name] = state
                            elif args.limited_lora == 0:
                                for targetname in v7_additional_parameters:
                                    if targetname in name and targetname != '' and name != '':
                                        lora_dict[name] = state
                                        break
                        else:
                            if '.moe_info' in name or '.moe_experts' in name:
                                lora_dict[name] = state
                                #print(f'Additional config saved {name}')
                    except (KeyError, AttributeError):
                        # キーが存在しない場合や、requires_gradプロパティがない場合の処理
                        #print(f'{name} not found')
                        pass



                try:
                    my_save(
                        args, trainer,
                        lora_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                    if args.state and args.state_output_mode != 0:
                        my_save(
                        args, trainer,
                        state_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}-state.pth",
                        )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {(trainer.my_epoch_loss):.6f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.my_pile_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.my_pile_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
