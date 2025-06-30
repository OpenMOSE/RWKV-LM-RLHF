#RWKVInfer with Flash-Linear-Attention
#2024 OpenMOSE

#2024.12.16 x070 Test Implement

#Implement 'Multi Recurrent State Sampling'

import asyncio
import queue
from dataclasses import dataclass,field
from typing import Dict, List, Optional
from enum import Enum
import torch
import copy
import gc
import copy
import time
import os
import re

#from rwkv6fla import PIPELINE, RWKV6 as RWKV_6
from rwkvengine.rwkvcore import RWKV_x
from rwkvengine.misc import PIPELINE


lock = asyncio.Lock()


class PromptStatus(Enum):
    QUEUED = 1
    PROCESSING = 2
    COMPLETED = 3
    FAILED = 4

@dataclass
class Prompt:
    id: int = -1
    prompts: str = ''
    status: PromptStatus = 3
    result: Optional[str] = None
    maxtokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.3
    endtoken: List[str] = field(default_factory=lambda: ['\n\n'])
    base_state_tuned: str = "None"
    use_exist_state_wkv: Optional[torch.Tensor] = None
    use_exist_state_shift: Optional[torch.Tensor] = None
    use_mrss: bool = False
    fixed_state_count: int = 0
    use_contain_originalstate: bool = False
    mrss_gating_param: List[float] = field(default_factory=lambda: [0.0,0.0,0.0,0.0])
    input_logits: List[torch.Tensor] = field(default_factory=list)
    input_logits_record:bool = False



class PromptQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.prompts: Dict[int, Prompt] = {}
        self.current_id = 0

    async def add_prompt(self, promptdata:Prompt) -> int:
        self.current_id += 1
        prompt = promptdata#Prompt(id=self.current_id, prompts=content, status=PromptStatus.QUEUED,temperature=)
        prompt.id=self.current_id
        prompt.status = PromptStatus.QUEUED
        self.prompts[prompt.id] = prompt
        await self.queue.put(prompt)
        return prompt.id

    async def get_prompt(self) -> Prompt:
        return await self.queue.get()

    def update_prompt(self, prompt_id: int,
                      status: PromptStatus,
                      result: Optional[str] = None, 
                      wkv_state : Optional[torch.Tensor]=None,
                      shift_state : Optional[torch.Tensor]=None, 
                       
                        ):
        if prompt_id in self.prompts:
            self.prompts[prompt_id].status = status
            if result is not None:
                self.prompts[prompt_id].result = result
            if wkv_state is not None:
                self.prompts[prompt_id].use_exist_state_wkv = wkv_state
            if shift_state is not None:
                self.prompts[prompt_id].use_exist_state_shift = shift_state

prompt_queue = PromptQueue()


class TextProcessor:
    def __init__(self, target="<RWKVArtifact"):
        self.target = target
        self.buffer = ""
        self.tag_buffer = ""
        self.is_in_tag = False

    def process_text(self, new_text):
        if not new_text:
            return "", None

        output = ""
        tag_output = None
        
        if self.is_in_tag:
            # タグ処理中の場合
            end_pos = new_text.find('>')
            if end_pos >= 0:
                # タグの終了を見つけた
                self.tag_buffer += new_text[:end_pos + 1]
                tag_output = self.tag_buffer
                self.is_in_tag = False
                self.tag_buffer = ""
                output = new_text[end_pos + 1:]  # タグ後のテキストは即時出力
            else:
                # タグが未完了
                self.tag_buffer += new_text
        else:
            # 通常テキスト処理中の場合
            process_text = self.buffer + new_text
            self.buffer = ""
            
            # targetの検索
            pos = process_text.find(self.target)
            
            if pos >= 0:
                # タグの開始を検出
                output = process_text[:pos]  # タグ前のテキストを出力
                self.is_in_tag = True
                self.tag_buffer = self.target
                remaining = process_text[pos + len(self.target):]
                
                # 残りのテキストでタグの終了を確認
                end_pos = remaining.find('>')
                if end_pos >= 0:
                    self.tag_buffer += remaining[:end_pos + 1]
                    tag_output = self.tag_buffer
                    self.is_in_tag = False
                    self.tag_buffer = ""
                    output += remaining[end_pos + 1:]  # タグ後のテキストも出力
                else:
                    self.tag_buffer += remaining
            else:
                # タグの部分一致をチェック
                for i in range(1, min(len(self.target) + 1, len(process_text) + 1)):
                    if self.target.startswith(process_text[-i:]):
                        # 部分一致を見つけた
                        output = process_text[:-i]
                        self.buffer = process_text[-i:]
                        break
                else:
                    # 部分一致もない場合は全て出力
                    output = process_text

        return output, tag_output
    
    def get_type_from_artifact(self,text):
        #try:
            # type属性の値を取得
            print('re search')
            type_value = re.search(r'language="([^"]*)"', text)
            print(type_value)
            if type_value:
                return type_value.group(1)
            else:
                # typeが見つからない場合はMarkdownを返す
                if 'html' in text:
                    return 'html'
                return "text/markdown"
        #except Exception as e:
        #    # エラーが発生した場合もMarkdownを返す
        #    print(f"エラーが発生しました: {e}")
        #    return "text/markdown"

class LLMWorker:
    def __init__(self,max_batch_size = 16):
        print('Initializing LLM Worker')
        
        self.llm_batch_chunk = 4096 #FLA Preprocess Prompt chunks
        self.llm_minimum_chunk = 128
        self.llm_batch_cycle = 10 #Preprocess cycle if 4, Pre,single,single,single,Pre,....
        self.llm_work_cycle = 0

        self.llm_max_batch_count = max_batch_size#16

        self.llm_last_batch_info = []
        self.llm_last_batch_count = 0
        self.llM_current_batch_info = []
        self.llm_current_batch_count = 0
        self.llm_dynamic_state_cache = []

        self.pipeline = PIPELINE()
        self.proceed_total_batches = 0
        
        self.ForceExit = False

        self.time_debug = False


        #async with lock:
        for i in range(self.llm_max_batch_count):
            data = {'slotstatus':'idle', # idle,preprocess,processing,finished
                    'prompt':[], #tokenized tokens
                    'prompt_id':-1, #queue prompt_id
                    'proceedtokens':0, 
                    'max_tokens': 200,
                    'temperature': 1.0,
                    'top_p' : 0.5,
                    'end_token': ['default'],
                    'remarks':'',
                    'use_state_tuned':'None',
                    'wkv_states' : None,
                    'shift_states' : None,
                    'output':'',
                    'currenttoken':None,
                    'currenttokencount': 0,
                    }
            self.llM_current_batch_info.append(data)
            self.llm_last_batch_info.append(data)





        

    def LoadModel(self,modelpath,quantize=False,precision='bf16',adapter_model = '',adapter_mode = '',adapter_scale=2.0, fully_fusedrecurrent = True):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        self.model = RWKV_x(modelpath,base_precision=precision,adapter_model=adapter_model,adapter_mode=adapter_mode,adapter_scale=adapter_scale,fully_fusedrecurrent=fully_fusedrecurrent)
        gc.collect()
        torch.cuda.empty_cache()
        print('model loaded')

    def UnloadModel(self):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print('model unloaded')
        gc.collect()
        torch.cuda.empty_cache()

    async def FLAGenerate(self,Queues:Prompt):
        global prompt_queue
        queue_id = await prompt_queue.add_prompt(Queues)
        currenttoken = ''
        Artifact = TextProcessor('<RWKVArtifact')
        Artifact2 = TextProcessor(target='</RWKVArtifact')
        while True:
            output = prompt_queue.prompts[queue_id].result
           # print(output)

            if output is not None:
                if len(output) > 0:
                    if len(currenttoken) < len(output):
                        splittext = output[len(currenttoken):]
                        splittext, tag = Artifact.process_text(splittext)
                        splittext, tag2 = Artifact2.process_text(splittext)
                        #print(f'tag = {tag}')
                        if tag is not None:
                            print('Artifact Detected. Analyzing')
                            typestyle = Artifact.get_type_from_artifact(tag)
                            print(f'gettype = {typestyle}')
                            splittext = f'```{typestyle}' + splittext
                        if tag2 is not None:
                            splittext = f'\n```' + splittext
                        currenttoken = output
                        yield splittext, None, None


            if prompt_queue.prompts[queue_id].status == PromptStatus.COMPLETED or prompt_queue.prompts[queue_id].status == PromptStatus.FAILED:
                yield "", copy.deepcopy(prompt_queue.prompts[queue_id].use_exist_state_wkv.to('cpu')), copy.deepcopy(prompt_queue.prompts[queue_id].use_exist_state_shift.to('cpu'))
                prompt_queue.prompts[queue_id] = None
                break
            if output is not None:
                if len(output) > 0:
                    await asyncio.sleep(0.001)
                else:
                    await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.1)

            




    async def QueueManage(self):
        global prompt_queue
        print('Start Queue Management')
        while True:
            if self.ForceExit:
                print('Queue Exit')
                return

            IdleSlot = 0
            async with lock:
                prompt = await prompt_queue.get_prompt()
                for i in range(self.llm_max_batch_count):
                    if self.llM_current_batch_info[i]['slotstatus'] == 'idle':
                        IdleSlot = IdleSlot + 1
                        #prompt = await prompt_queue.get_prompt()
                        #print(prompt)
                        prompt_queue.update_prompt(prompt.id, PromptStatus.PROCESSING)
                        self.llM_current_batch_info[i]['slotstatus'] = 'processing'

                        #print('/////////////////////////////////////////////////')
                        #print(prompt)

                        data = {'slotstatus':'processing', # idle,preprocess,processing,finished
                                'prompt':self.pipeline.encode(prompt.prompts), #tokenized tokens
                                'prompt_id':prompt.id, #queue prompt_id
                                'proceedtokens':0, 
                                'max_tokens': prompt.maxtokens,
                                'currenttokencount': 0,
                                'temperature': prompt.temperature,
                                'top_p' : prompt.top_p,
                                'end_token': prompt.endtoken,
                                'remarks':f'{str(self.proceed_total_batches)}',
                                'use_state-tuned':prompt.base_state_tuned,
                                'wkv_states' : prompt.use_exist_state_wkv,
                                'shift_states' : prompt.use_exist_state_shift,
                                'current_prob' : None,
                                'input_logits' : [],
                                'input_logits_record' : prompt.input_logits_record,
                                'output':'',
                                'out_tokens':[],
                                'out_last':0,
                                'currenttoken':self.pipeline.encode(''),
                                'occurrence':{},
                                'count':0,
                                'start_time':None,
                                'end_time':None,
                                #MRSS
                                'use_contain_originalstate':prompt.use_contain_originalstate,
                                'use_mrss':prompt.use_mrss,
                                'mrss_gating_param':prompt.mrss_gating_param,
                                'mrss_state_count': 0,
                                'fixed_state_count':prompt.fixed_state_count
                                }
                        
                        #print(data)
                        
                        self.llM_current_batch_info[i] = data#copy.deepcopy(data)
                        del data
                        
                        self.proceed_total_batches = self.proceed_total_batches + 1
                        if self.proceed_total_batches > 2000000000:
                            self.proceed_total_batches = 0
                        break
                    

                    
            
            await asyncio.sleep(0.1) # Everyone 10ms loop


    async def RunLLM(self):
        print('Start LLM Engine')
        global prompt_queue
        while True:
            if self.ForceExit:
                print('LLM Exit')
                return
            NoProcessing = 1
            for i in range(self.llm_max_batch_count):
                if self.llM_current_batch_info[i]['slotstatus'] == 'processing':
                    NoProcessing = 0
            if NoProcessing == 0:
                #print('have work :)')
                self.llm_work_cycle = self.llm_work_cycle + 1
                if self.llm_work_cycle > 1000:
                    self.llm_work_cycle = 0

                if self.llm_work_cycle % self.llm_batch_cycle == 0:
                    #batch Pre Process Mode:
                    #print('pre processing mode')
                    #check
                    prompts = []
                    prompts_ids = [] 
                    b_wkv_states = []
                    b_shift_states = []
                    mrss_info = []
                    input_logits = []
                    input_logits_record = []
                    token_max = self.llm_batch_chunk
                    #for work in self.llM_current_batch_info:
                    #    if work['proceedtokens'] < len(work['prompt']):
                    #        if len(work['prompt']) - work['proceedtokens'] < token_max:
                    #            token_max = len(work['prompt']) - work['proceedtokens']
                    for i in range(self.llm_max_batch_count):
                        work = self.llM_current_batch_info[i]
                        
                        if work['proceedtokens'] < len(work['prompt']) and work['slotstatus'] == 'processing':
                            print(f"batch {i} input tokens = {len(work['prompt'])}")
                            if len(work['prompt']) - work['proceedtokens'] < token_max:
                                token_max = len(work['prompt']) - work['proceedtokens']

                    if token_max <= self.llm_minimum_chunk:
                        token_max = token_max  
                    else:
                        token_max = (token_max // self.llm_minimum_chunk) * self.llm_minimum_chunk
                                
                    for i in range(self.llm_max_batch_count):
                        work = self.llM_current_batch_info[i]
                        if work['proceedtokens'] < len(work['prompt']) and work['slotstatus'] == 'processing':
                                prompts.append(work['prompt'][work['proceedtokens']:work['proceedtokens']+token_max])
                                prompts_ids.append(work['prompt_id'])
                                b_wkv_states.append(work['wkv_states'])
                                b_shift_states.append(work['shift_states'])
                                input_logits.append(work['input_logits'])
                                input_logits_record.append(work['input_logits_record'])
                                mrss_info.append({'use_contain_originalstate':work['use_contain_originalstate'], # True or False
                                                  'use_mrss':work['use_mrss'], # True or False
                                                  'mrss_gating_param':work['mrss_gating_param'], # gating params list
                                                  'mrss_state_count':work['mrss_state_count'],
                                                  'fixed_state_count':work['fixed_state_count']
                                                  })

                    #print(f'prompts = {prompts}')

                    #print(f'prompts_decoded = {self.pipeline.decode(prompts[0])}')
                    
                    if len(prompts)>0: # if have pre process work
                        prompts_tensor = []
                        #for tx in prompts:
                        #    prompts_tensor.append(torch.tensor(tx).unsqueeze(0).to('cuda'))

                        realbatchcount = 0

                        for i in range(len(prompts)):
                            if mrss_info[i]['use_mrss'] == True:
                                #Use MRSS Mode
                                mrss_state_count = mrss_info[i]['fixed_state_count']#len(b_wkv_states[i])
                                realbatchcount = realbatchcount + mrss_state_count
                                localbatchcount = mrss_state_count
                                for j in range(mrss_state_count):
                                    prompts_tensor.append(torch.tensor(prompts[i]).unsqueeze(0).to('cuda'))
                                if mrss_info[i]['use_contain_originalstate'] == True:
                                    prompts_tensor.append(torch.tensor(prompts[i]).unsqueeze(0).to('cuda'))
                                    realbatchcount = realbatchcount + 1
                                    localbatchcount = localbatchcount + 1
                                mrss_info[i]['mrss_state_count'] = localbatchcount
                            else:
                                realbatchcount = realbatchcount + 1
                                prompts_tensor.append(torch.tensor(prompts[i]).unsqueeze(0).to('cuda'))
                        
                        #print(f'realbatchcount = {realbatchcount}')

                        idx = torch.cat(prompts_tensor, dim=0) # same realbatchcount
        
                        self.States = self.model.new_state(realbatchcount)

                        if self.model.RWKVMode == 6:
                            shift_states = self.States.shift_states.permute(1, 0, 2, 3)
                            wkv_states = self.States.wkv_states.permute(1, 0, 2, 3, 4)
                        elif self.model.RWKVMode == 7:
                            shift_states = self.States.shift_states.permute(1, 0, 2)
                            wkv_states = self.States.wkv_states.permute(1, 0, 2, 3, 4)

                        NowTensorPosition = 0
                        for i in range(len(prompts)):
                            if mrss_info[i]['use_mrss'] == True:
                                print('MRSS Mode')
                                if type(b_wkv_states[i])==list:
                                    b_wkv_states[i] = torch.stack(b_wkv_states[i],dim=0)

                                mrss_state_count = mrss_info[i]['fixed_state_count']#len(b_wkv_states[i])
                                for j in range(mrss_state_count):
                                    wkv_states[NowTensorPosition + j] = b_wkv_states[i][j]

                                if mrss_info[i]['use_contain_originalstate'] == True:
                                    if len(b_wkv_states[i]) == mrss_state_count + 1:
                                        wkv_states[NowTensorPosition + mrss_state_count] = b_wkv_states[i][mrss_state_count]
                                    #else:
                                    #    print('through')

                                if b_shift_states[i] is not None:
                                    if type(b_shift_states[i])==list:
                                        b_shift_states[i] = torch.stack(b_shift_states[i],dim=0)
                                    for j in range(mrss_state_count):
                                        shift_states[NowTensorPosition + j] = b_shift_states[i][j]

                                NowTensorPosition = NowTensorPosition + mrss_state_count
                                if mrss_info[i]['use_contain_originalstate'] == True:
                                    NowTensorPosition = NowTensorPosition + 1

                            else:
                                #print('Normal Mode')
                                if b_wkv_states[i] is not None:
                                    if type(b_wkv_states[i])==list:
                                        b_wkv_states[i] = torch.stack(b_wkv_states[i],dim=0)
                                    wkv_states[NowTensorPosition] = b_wkv_states[i]

                                if b_shift_states[i] is not None:
                                    if type(b_shift_states[i])==list:
                                        b_shift_states[i] = torch.stack(b_shift_states[i],dim=0)
                                    shift_states[NowTensorPosition] = b_shift_states[i]
                                
                                NowTensorPosition = NowTensorPosition + 1


                        if self.model.RWKVMode == 6:
                            shift_states = shift_states.permute(1,0,2,3) 
                            wkv_states = wkv_states.permute(1, 0, 2, 3, 4) 
                        elif self.model.RWKVMode == 7:
                            shift_states = shift_states.permute(1,0,2) 
                            wkv_states = wkv_states.permute(1, 0, 2, 3, 4) 


                        if token_max < self.llm_minimum_chunk:
                            KernelMode = 2
                        else:
                            KernelMode = 0

                        x, shift_states, wkv_states = self.model.forward(idx, shift_states, wkv_states,KernelMode=KernelMode)

                        

                        if x.dim() == 2:
                            x = x.view(x.shape[0],1,x.shape[1])
                        print(f'x shape = {x.shape}')
                        print(f'{token_max} forwarded')

                        #print(f'x = {x}')
                        #print(f'x.shape = {x.shape}')
                        if self.model.RWKVMode == 6:
                            shift_states = shift_states.permute(1,0,2,3) 
                            wkv_states = wkv_states.permute(1, 0, 2, 3, 4) 
                        elif self.model.RWKVMode == 7:
                            shift_states = shift_states.permute(1,0,2) 
                            wkv_states = wkv_states.permute(1, 0, 2, 3, 4) 

                        j = -1
                        NowTensorPosition = 0
                        for id in prompts_ids:
                            j = j + 1
                            for i in range(self.llm_max_batch_count):
                                if self.llM_current_batch_info[i]['prompt_id'] == id:
                                    if mrss_info[j]['use_mrss'] == True:
                                        mrss_state_count = mrss_info[j]['mrss_state_count']
                                        self.llM_current_batch_info[i]['wkv_states'] = wkv_states[NowTensorPosition:(NowTensorPosition+mrss_state_count)]
                                        self.llM_current_batch_info[i]['shift_states'] = shift_states[NowTensorPosition:(NowTensorPosition+mrss_state_count)]
                                        self.llM_current_batch_info[i]['current_prob'] = x[NowTensorPosition:(NowTensorPosition+mrss_state_count)]
                                        self.llM_current_batch_info[i]['proceedtokens'] = self.llM_current_batch_info[i]['proceedtokens'] + token_max
                                        self.llM_current_batch_info[i]['mrss_state_count'] = mrss_info[j]['mrss_state_count']
                                        NowTensorPosition = NowTensorPosition + mrss_state_count
                                    else:
                                        self.llM_current_batch_info[i]['wkv_states'] = wkv_states[NowTensorPosition]
                                        self.llM_current_batch_info[i]['shift_states'] = shift_states[NowTensorPosition]
                                        self.llM_current_batch_info[i]['current_prob'] = x[NowTensorPosition]
                                        self.llM_current_batch_info[i]['proceedtokens'] = self.llM_current_batch_info[i]['proceedtokens'] + token_max
                                        NowTensorPosition = NowTensorPosition + 1

                else:
                    #print('recurrent infer mode')
                    #prompts = []
                    if self.time_debug:
                        start_time = time.time()
                    token = []
                    token_ids = [] 
                    b_wkv_states = []
                    b_shift_states = []
                    current_prob = []
                    temperature = []
                    top_p = []
                    outputs = []
                    out_tokens = []
                    out_last = []
                    max_tokens = []
                    statuss = []
                    end_token = []
                    occurrence = []
                    counts = []
                    mrss_info = []

                    start_times = []
                    end_times = []

                

                    # フィルタリング条件
                    valid_works = [work for work in self.llM_current_batch_info[:self.llm_max_batch_count]
                                if work['proceedtokens'] >= len(work['prompt']) and work['slotstatus'] == 'processing']
                    
                    #print(valid_works)

                    # リスト内包表記を使用してデータを抽出
                    token = [work['currenttoken'] for work in valid_works]
                    token_ids = [work['prompt_id'] for work in valid_works]
                    b_wkv_states = [work['wkv_states'] for work in valid_works]
                    b_shift_states = [work['shift_states'] for work in valid_works]
                    current_prob = [work['current_prob'] for work in valid_works]
                    temperature = [torch.Tensor([float(work['temperature'])]) for work in valid_works]
                    top_p = [torch.Tensor([float(work['top_p'])]) for work in valid_works]
                    outputs = [work['output'] for work in valid_works]
                    out_tokens = [work['out_tokens'] for work in valid_works]
                    out_last = [work['out_last'] for work in valid_works]
                    max_tokens = [work['max_tokens'] for work in valid_works]
                    statuss = [work['slotstatus'] for work in valid_works]
                    end_token = [work['end_token'] for work in valid_works]
                    occurrence = [work['occurrence'] for work in valid_works]
                    counts = [work['count'] for work in valid_works]
                    start_times = [work['start_time'] for work in valid_works]
                    end_times = [work['end_time'] for work in valid_works]
                    mrss_info = [{'use_contain_originalstate': work['use_contain_originalstate'],
                                'use_mrss': work['use_mrss'],
                                'mrss_gating_param': work['mrss_gating_param'],
                                'mrss_state_count': work['mrss_state_count']}
                                for work in valid_works]

                    if self.time_debug:
                        start_time1 = time.time()

                    if len(token_ids) > 0:
                        otokens = []
                        NowRealBatchPosition = 0
                        realbatchcount = 0


                        #New Implement MultiBatch Sampling

                        BatchProbs = []
                        for j in range(len(token_ids)):
                            if start_times[j] is None:
                                 start_times[j] = time.time()

                            if mrss_info[j]['use_mrss'] == True: #MRSS mode
                                realbatchcount = realbatchcount + mrss_info[j]['mrss_state_count']
                                mrss_state_count = mrss_info[j]['mrss_state_count']
                                
                                if len(mrss_info[j]['mrss_gating_param']) <  mrss_state_count:
                                    current_gating_param_count  = len(mrss_info[j]['mrss_gating_param'])
                                    for k in range(mrss_state_count - current_gating_param_count):
                                        mrss_info[j]['mrss_gating_param'].append(0.0) #add dummy gating weight
                                #print(f"Current GatingWeightCount = {len(mrss_info[j]['mrss_gating_param'])}")

                                logits_combined = None
                                totalweight = 0.0
                                for k in range(mrss_state_count):
                                    totalweight = totalweight + mrss_info[j]['mrss_gating_param'][k]
                                if totalweight == 0.0:
                                    totalweight = 1.0
                                #print(f'mrss_state_count = {mrss_state_count}')
                                for k in range(mrss_state_count):
                                    for n in occurrence[j]:
                                        current_prob[j][k][-1][n] -= 0.2 + occurrence[j][n] * 0.3
                                    #print(f'current_prob[j] length = {len(current_prob[j])}')
                                    current_prob[j][k][-1][0] -= 1e10
                                    if logits_combined is None:
                                        if mrss_info[j]['mrss_gating_param'][k] != 0:
                                            logits_combined = current_prob[j][k][-1] * (mrss_info[j]['mrss_gating_param'][k] / totalweight)
                                        #totalweight = totalweight + mrss_info[j]['mrss_gating_param'][k]
                                    else:
                                        if mrss_info[j]['mrss_gating_param'][k] != 0:
                                            logits_combined = logits_combined + current_prob[j][k][-1] * (mrss_info[j]['mrss_gating_param'][k] / totalweight)
                                        #totalweight = totalweight + mrss_info[j]['mrss_gating_param'][k]

                                #logits_combined = logits_combined / totalweight
                                BatchProbs.append(logits_combined)
                            else:
                                realbatchcount = realbatchcount + 1

                                for n in occurrence[j]:
                                    current_prob[j][-1][n] -= 0.2 + occurrence[j][n] * 0.3

                                current_prob[j][-1][0] -= 1e10 

                                BatchProbs.append(current_prob[j][-1])
                        
                        #Batch Sampling
                        #BatchProbs[:, 0] -= 1e10
                        BatchProbs = torch.stack(BatchProbs)
                        #print(temperature)
                        otokens = self.pipeline.improved_nucleus_sampling_multi_static(BatchProbs, temperature=torch.stack(temperature), top_p=torch.stack(top_p)).tolist()

                        for j in range(len(token_ids)):
                            for xxx in occurrence[j]:
                                    occurrence[j][xxx] *= 0.996
                            tk = otokens[j]
                            occurrence[j][tk] = 1 + (occurrence[j][tk] if tk in occurrence[j] else 0)

  





                         



                        tokens = []
                        for j in range(len(token_ids)):
                            if mrss_info[j]['use_mrss'] == True: #MRSS mode
                                mrss_state_count = mrss_info[j]['mrss_state_count']
                                for k in range(mrss_state_count):
                                    tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0))#.to('cuda'))
                            else:
                                tokens.append(torch.tensor(otokens[j]).unsqueeze(0).unsqueeze(0))#.to('cuda'))

                        for j in range(len(token_ids)):
                            out_tokens[j] += [otokens[j]]
                            try:
                                tmp = self.pipeline.decode(out_tokens[j][out_last[j]:])
                                if ("\ufffd" not in tmp) and (not tmp.endswith("\n")) and (not tmp.endswith(end_token[j][0])):
                                        #yield tmp
                                        #if j == Target_batch - 1:
                                        #print(tmp,end="", flush=True)
                                        #if j == 0:
                                        #    print(tmp,end="", flush=True)
                                        #print(tmp,end="", flush=True)
                                        outputs[j] = outputs[j] + tmp
                                        out_last[j] = counts[j] + 1
                                #print(f'outtokens = {len(out_tokens[j])}')
                                #print(f'{int(counts[j])} {max_tokens[j]}')
                                if int(counts[j]) > int(max_tokens[j]):
                                    #Reached Max Token
                                    statuss[j] = 'idle'
                                    print(f'batch {j} is reached max_tokens')
                                    #print(outputs[j])
                                if len(out_tokens[j]) > int(max_tokens[j]):
                                    #Reached Max Token
                                    statuss[j] = 'idle'
                                    print(f'batch {j} is finished')
                                    #print(outputs[j])
                                exit_flag = False
                                #print(end_token[j])
                                for stop in end_token[j]:
                                    if stop in tmp:
                                        #yield tmp
                                        #output_text = output_text + tmp
                                        print(f'Endtoken = {repr(tmp)}')
                                        tmp = tmp.replace(stop,'')
                                        #outputs[j] = outputs[j] + tmp
                                        exit_flag = True
                                if exit_flag:
                                    statuss[j] = 'idle'
                                    outputs[j] = outputs[j] + tmp
                                    print(f'batch {j} is finished cause got endtoken')
                                    #print(outputs[j])

                            except Exception as e:
                                print('exceptions')
                                #print(f"エラーが発生しました: {type(e).__name__}")
                                #print(f"エラーの詳細: {str(e)}")
                                #print(f'tried tokenize {out_tokens[j][out_last[j]:]}')
                                #print(f'tried tokenize {out_tokens[j]}')
                                #tmp = ''
                                outputs[j] = outputs[j] #+ tmp
                                #out_last[j] = counts[j] + 1
                                pass

                        if self.time_debug:
                            start_time2 = time.time()

                        idx = torch.cat(tokens, dim=0).to('cuda')

                        self.States = self.model.new_state((realbatchcount))

                        #print(f'realbatchcount={realbatchcount}')
                        if self.model.RWKVMode == 6:
                            shift_states = self.States.shift_states.permute(1, 0, 2, 3) 
                            wkv_states = self.States.wkv_states.permute(1, 0, 2, 3, 4) 
                        elif self.model.RWKVMode == 7:
                            shift_states = self.States.shift_states.permute(1, 0, 2) 
                            wkv_states = self.States.wkv_states.permute(1, 0, 2, 3, 4) 

                        # for i in range(len(token)):
                        #     if b_wkv_states[i] is not None:
                        #         wkv_states[i] = b_wkv_states[i]#prompts['wkv_states']
                        #     #else:
                        #     #    print('wkv is none')
                        #     if b_shift_states[i] is not None:
                        #         shift_states[i] = b_shift_states[i]#prompts['shift_states']
                        #     #else:
                        #     #    print('shift is none')
                        NowTensorPosition = 0
                        for i in range(len(token_ids)):
                            if mrss_info[i]['use_mrss'] == True:
                                #print('MRSS Mode')
                                if type(b_wkv_states[i])==list:
                                    b_wkv_states[i] = torch.stack(b_wkv_states[i],dim=0)

                                mrss_state_count = len(b_wkv_states[i])
                                for j in range(mrss_state_count):
                                    wkv_states[NowTensorPosition + j] = b_wkv_states[i][j]

                                if mrss_info[i]['use_contain_originalstate'] == True:
                                    if len(b_wkv_states[i]) == mrss_state_count + 1:
                                        wkv_states[NowTensorPosition + mrss_state_count] = b_wkv_states[i][mrss_state_count]
                                    #else:
                                    #    print('through')

                                if b_shift_states[i] is not None:
                                    if type(b_shift_states[i])==list:
                                        b_shift_states[i] = torch.stack(b_shift_states[i],dim=0)
                                    for j in range(mrss_state_count):
                                        shift_states[NowTensorPosition + j] = b_shift_states[i][j]

                                NowTensorPosition = NowTensorPosition + mrss_state_count
                                #if mrss_info[i]['use_contain_originalstate'] == True and mrss_state_count != mrss_info[i]['fixed_state_count']:
                                #    NowTensorPosition = NowTensorPosition + 1

                            else:
                                #print('Normal Mode')
                                if b_wkv_states[i] is not None:
                                    if type(b_wkv_states[i])==list:
                                        b_wkv_states[i] = torch.stack(b_wkv_states[i],dim=0)
                                    wkv_states[NowTensorPosition] = b_wkv_states[i]

                                if b_shift_states[i] is not None:
                                    if type(b_shift_states[i])==list:
                                        b_shift_states[i] = torch.stack(b_shift_states[i],dim=0)
                                    shift_states[NowTensorPosition] = b_shift_states[i]
                                
                                NowTensorPosition = NowTensorPosition + 1
                        if self.model.RWKVMode == 6:
                            shift_states = shift_states.permute(1,0,2,3)
                            wkv_states = wkv_states.permute(1, 0, 2, 3, 4)
                        elif self.model.RWKVMode == 7:
                            shift_states = shift_states.permute(1,0,2)
                            wkv_states = wkv_states.permute(1, 0, 2, 3, 4)

                        x, shift_states, wkv_states = self.model.forward(idx, shift_states, wkv_states,one_mode=True)

                        if x.dim() == 2:
                            x = x.view(x.shape[0],1,x.shape[1])

                        if self.time_debug:
                            start_time3 = time.time()

                        #print(f'probs shape = {x.shape}')

                        if self.model.RWKVMode == 6:
                            shift_states = shift_states.permute(1,0,2,3)
                            wkv_states = wkv_states.permute(1, 0, 2, 3, 4)
                        elif self.model.RWKVMode == 7:
                            shift_states = shift_states.permute(1,0,2)
                            wkv_states = wkv_states.permute(1, 0, 2, 3, 4)

                        j = -1
                        NowTensorPosition = 0
                        for id in token_ids:
                            j = j + 1
                            for i in range(self.llm_max_batch_count):
                                if self.llM_current_batch_info[i]['prompt_id'] == id:

                                    #print('waiting lock')

                                    
                                        #print('locked')

                                    if mrss_info[j]['use_mrss'] == True:
                                        mrss_state_count = mrss_info[j]['mrss_state_count']
                                        if statuss[j] == 'processing':
                                            prompt_queue.update_prompt(id,PromptStatus.PROCESSING,result=outputs[j])
                                            self.llM_current_batch_info[i]['wkv_states'] = wkv_states[NowTensorPosition:(NowTensorPosition+mrss_state_count)]
                                            self.llM_current_batch_info[i]['shift_states'] = shift_states[NowTensorPosition:(NowTensorPosition+mrss_state_count)]
                                            self.llM_current_batch_info[i]['current_prob'] = x[NowTensorPosition:(NowTensorPosition+mrss_state_count)]
                                            self.llM_current_batch_info[i]['occurrence'] = occurrence[j]
                                        else:
                                            if end_times[j] is None:
                                                end_times[j] = time.time()
                                                duration =  end_times[j] - start_times[j]
                                                tokencount = len(out_tokens[j])
                                                token_performance = tokencount / duration
                                                print(f'batch{i} : finished. {token_performance:0.2f} t/s')

                                            prompt_queue.update_prompt(id,PromptStatus.COMPLETED,result=outputs[j],wkv_state=wkv_states[NowTensorPosition:(NowTensorPosition+mrss_state_count)].to('cpu'),shift_state=shift_states[NowTensorPosition:(NowTensorPosition+mrss_state_count)].to('cpu'))
                                            self.llM_current_batch_info[i]['wkv_states'] = None
                                            self.llM_current_batch_info[i]['shift_states'] = None
                                            self.llM_current_batch_info[i]['current_prob'] = None
                                            self.llM_current_batch_info[i]['occurrence'] = None
                                            statuss[j] == 'idle'
                                        
                                        

                                        NowTensorPosition = NowTensorPosition + mrss_state_count

                                    else:
                                        if statuss[j] == 'processing':
                                            prompt_queue.update_prompt(id,PromptStatus.PROCESSING,result=outputs[j])

                                            self.llM_current_batch_info[i]['wkv_states'] = wkv_states[NowTensorPosition]
                                            self.llM_current_batch_info[i]['shift_states'] = shift_states[NowTensorPosition]
                                            self.llM_current_batch_info[i]['current_prob'] = x[NowTensorPosition]
                                            self.llM_current_batch_info[i]['occurrence'] = occurrence[j]
                                        else:
                                            if end_times[j] is None:
                                                end_times[j] = time.time()
                                                duration =  end_times[j] - start_times[j]
                                                tokencount = len(out_tokens[j])
                                                token_performance = tokencount / duration
                                                print(f'batch{i} : finished. {token_performance:0.2f} t/s')

                                            prompt_queue.update_prompt(id,PromptStatus.COMPLETED,result=outputs[j],wkv_state=wkv_states[NowTensorPosition].to('cpu'),shift_state=shift_states[NowTensorPosition].to('cpu'))
                                            statuss[j] == 'idle'

                                            self.llM_current_batch_info[i]['wkv_states'] = None
                                            self.llM_current_batch_info[i]['shift_states'] = None
                                            self.llM_current_batch_info[i]['current_prob'] = None
                                            self.llM_current_batch_info[i]['occurrence'] = None

                                        NowTensorPosition = NowTensorPosition + 1



                                    self.llM_current_batch_info[i]['currenttoken'] = otokens[j]
                                    self.llM_current_batch_info[i]['slotstatus'] = statuss[j]

                                    self.llM_current_batch_info[i]['start_time'] = start_times[j]
                                    self.llM_current_batch_info[i]['end_time'] = end_times[j]

                                    self.llM_current_batch_info[i]['output'] = outputs[j]
                                    self.llM_current_batch_info[i]['out_tokens'] = out_tokens[j]
                                    self.llM_current_batch_info[i]['out_last'] = out_last[j]
                                    
                                    self.llM_current_batch_info[i]['count'] = counts[j] + 1
                                    #self.llM_current_batch_info[i]['mrss_state_count'] = mrss_info[j]['mrss_state_count']
                                    break
                        if self.time_debug:
                            start_time4 = time.time()

                        if self.time_debug:
                            StoreTime = start_time4 - start_time3
                            InferenceTime = start_time3 - start_time2
                            DecodeTime = start_time2 - start_time1
                            FetchTime = start_time1 - start_time
                            #SamplerTime = start_time_sample1 - start_time_sample

                            print(f'FetchTime = {FetchTime*1000:0.4f}')
                            #print(f'SamplerTime = {SamplerTime*1000:0.4f}')
                            #print(f'DecodeTime = {DecodeTime*1000:0.4f}')
                            print(f'InferenceTime = {InferenceTime*1000:0.4f}')
                            print(f'StoreTime = {StoreTime*1000:0.4f}')
            


                    

                    




            await asyncio.sleep(0.00001) # Every 1ms
