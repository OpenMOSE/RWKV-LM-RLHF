import os, math, time, datetime, subprocess
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

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

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        
        #print(0)
        torch.cuda.empty_cache()

        np.random.seed(trainer.global_step + args.lisa_rand_seed)
        
        if args.lisa:

            if args.lisa_plus_enabled == 1:
                self.att_elements = args.lisa_plus_att_train_params.split(',')
                self.att_number_of_elements = len(self.att_elements)
                self.ffn_elements = args.lisa_plus_ffn_train_params.split(',')
                self.ffn_number_of_elements = len(self.ffn_elements)

                self.lisa_plus_att_permanent_freeze_params = args.lisa_plus_att_permanent_freeze_params.split(',')
                self.lisa_plus_att_permanent_freeze_params_number_of_elements = len(self.lisa_plus_att_permanent_freeze_params)
                self.lisa_plus_ffn_permanent_freeze_params = args.lisa_plus_ffn_permanent_freeze_params.split(',')
                self.lisa_plus_ffn_permanent_freeze_params_number_of_elements = len(self.lisa_plus_ffn_permanent_freeze_params)

                #print(f'self.lisa_plus_att_permanent_freeze_params_number_of_elements={self.lisa_plus_att_permanent_freeze_params_number_of_elements}')
                # Permanent Freeze リストを含む要素を削除
                if self.lisa_plus_att_permanent_freeze_params_number_of_elements > 0 and args.lisa_plus_att_permanent_freeze_params != '':
                    self.att_elements = [item for item in self.att_elements if all(s not in item for s in self.lisa_plus_att_permanent_freeze_params)]
                    #print(self.att_elements)
                    self.att_number_of_elements = len(self.att_elements)
                    if self.att_number_of_elements < args.lisa_plus_att_active_weight:
                        args.lisa_plus_att_active_weight = self.att_number_of_elements
                    #print(self.att_number_of_elements)

                if self.lisa_plus_ffn_permanent_freeze_params_number_of_elements > 0 and args.lisa_plus_ffn_permanent_freeze_params != '':
                    self.ffn_elements = [item for item in self.ffn_elements if all(s not in item for s in self.lisa_plus_ffn_permanent_freeze_params)]
                    self.ffn_number_of_elements = len(self.ffn_elements)
                    if self.ffn_number_of_elements < args.lisa_plus_ffn_active_weight:
                        args.lisa_plus_ffn_active_weight = self.ffn_number_of_elements
                    #print(self.ffn_elements)

                if args.lisa_plus_custom_layer_probabilities:
                    #Custom Layer Probabilities Mode Enabled
                    # CSVファイルのパス
                    csv_file_path = args.lisa_plus_custom_layer_probabilities_profile
                    # CSVファイルを読み込む
                    df = pd.read_csv(csv_file_path)
                    # 'Position'列を整数に変換
                    df['Position'] = df['Position'].astype(int)
                    # 'Value'列を浮動小数点数に変換
                    df['Value'] = df['Value'].astype(float)
                    # 空の辞書を作成
                    profile = []
                    # 辞書にデータを格納
                    i=0
                    for index, row in df.iterrows():
                        profile.append(row['Value'])
                        i=i+1
                    #Initialize probabilities
                    probabilities = np.ones(args.n_layer)
                    if args.n_layer != len(profile):
                        #Miss match profile maybe Wrong Profile Length or wrong n_layer
                        raise "The number of profile layers and the number of model layers do not match. Please check it"
                    
                    for i in range(0,args.n_layer):
                        probabilities[i]=profile[i]
                        #print(f'Layer {i} is now probability value {profile[i]}')
                    
                    probabilities /= probabilities.sum()  # 確率の合計が1になるように正規化
                else:
                    probabilities = np.ones(args.n_layer)
                    probabilities /= probabilities.sum()  # 確率の合計が1になるように正規化

                    



                #print(self.att_elements)


            if batch_idx % args.lisa_interval_steps == 0:  # 例として、10バッチごとに凍結
                for name, param in pl_module.named_parameters():
                    if 'blocks' in name:
                        param.requires_grad = False
                        param.grad = None
                        #print(f"Freezed: {name}")  # 凍結したパラメータの名前を表示
                if args.lisa_plus_enabled == 1:
                    #print(f'np random self.att_number_of_elements={self.att_number_of_elements} , args.lisa_plus_att_active_weight={args.lisa_plus_att_active_weight}')
                    self.att_disable_elements_indices = np.random.choice(range(self.att_number_of_elements),
                                                                 self.att_number_of_elements-args.lisa_plus_att_active_weight, replace=False)
                    self.ffn_disable_elements_indices = np.random.choice(range(self.ffn_number_of_elements),
                                                                 self.ffn_number_of_elements-args.lisa_plus_ffn_active_weight, replace=False)
                
                self.active_layers_indices = np.random.choice(range(args.n_layer), args.lisa_active_layer, replace=False,p=probabilities)
                for idx in self.active_layers_indices:
                        if args.lisa_debug == 1:
                            print(f'Now Activate Layer is {idx}')
                        for name, param in pl_module.named_parameters():
                            if 'blocks' in name and f'.{str(idx)}.' in name:
                                param.requires_grad = True
                                #print(name)

                                #print(f'self.att_disable_elements_indices = {self.att_disable_elements_indices}')

                                if args.lisa_plus_enabled == 1:
                                    for idx2 in self.att_disable_elements_indices:
                                        #print(f'{self.att_elements[idx2]} in name {name}')
                                        if f'{self.att_elements[idx2]}' in name:
                                            param.requires_grad = False
                                            param.grad=None
                                            if args.lisa_debug == 1:
                                                print(f'Now Layer {name} Element {self.att_elements[idx2]} is Disabled')
                                    for idx2 in self.ffn_disable_elements_indices:
                                        if f'{self.ffn_elements[idx2]}' in name:
                                            param.requires_grad = False
                                            param.grad=None
                                            if args.lisa_debug == 1:
                                                print(f'Now Layer {name} Element {self.ffn_elements[idx2]} is Disabled')   
                                    for FreezeName in self.lisa_plus_att_permanent_freeze_params:
                                        if FreezeName in name and FreezeName != '':
                                            param.requires_grad = False
                                            param.grad=None
                                            if args.lisa_debug == 1:
                                                print(f'Now Layer {name} Element {FreezeName} is Disabled Permanently') 
                                    for FreezeName in self.lisa_plus_ffn_permanent_freeze_params:
                                        if FreezeName in name and FreezeName != '':
                                            param.requires_grad = False
                                            param.grad=None
                                            if args.lisa_debug == 1:
                                                print(f'Now Layer {name} Element {FreezeName} is Disabled Permanently') 


                                #print(f'Now Layer {name} is enabled')
        
        
        #print(1)
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
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
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)
        #print(3)
        if args.my_exit_tokens != 0: # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init                
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):
                    my_save(
                        args, trainer,
                        pl_module.state_dict(),
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                    exit(0)
        #print(4)
        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)
        #print(5)
        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay
        #print(6)
        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
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
                    print("Login to wandb...")
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
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                if args.dpo:
                    try:
                        lll |= {"pref_percentage": trainer.pref_match_percentage, "loss_1": trainer.loss_1_general_or_sft, "loss_2_dpo": trainer.loss_2_dpo}
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
        assert "MyDataset" in str(dataset)
        if args.dpo:
            dataset[0].global_rank = trainer.global_rank
            dataset[0].real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset[0].world_size = trainer.world_size
            dataset[1].global_rank = trainer.global_rank
            dataset[1].real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset[1].world_size = trainer.world_size
        else:
            dataset.global_rank = trainer.global_rank
            dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
            dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
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
