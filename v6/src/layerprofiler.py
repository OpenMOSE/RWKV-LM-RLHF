#2024 OpenMOSE
import csv

class LayerProfiler:
    def __init__(self,filename_profile):
        def read_csv_to_array(file_path):
            data = []
            with open(file_path, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                for row in csv_reader:
                    data.append(row)
            return data
        
        array_data = read_csv_to_array(filename_profile)

        self.array_data = array_data

    def get_layer_info(self,layer_name):
        for row in self.array_data:
            if row['Layer'] == layer_name:
                Rt = {'Layer':row['Layer'],
                      'Mode':row['Mode'],
                      'Rank':row['Rank'],
                      'Alpha':row['Alpha'],
                      'LISAProb':row['LISAProb'],
                      'LoRAProb':row['LoRAProb'],
                      'LRScale':row['LRScale'],              
                      }
                return Rt
        return None
    def make_layer_config(self,nlayer,quant):
        CONFIG = {}
        for i in range(nlayer):
            Found = False
            for row in self.array_data:
                
                if row['Layer'] == str(i):
                    Found = True
                    if row['Mode'] == 'full':
                        CONFIG[f'{str(i)}']={'mode':'full'}
                    elif row['Mode'] == 'freeze':
                        CONFIG[f'{str(i)}']={'mode':'freeze', 'quant':quant }
                    elif row['Mode'] == 'lora':
                        CONFIG[f'{str(i)}']={'mode':'lora','quant':quant,'rank':row['Rank'],'alpha':row['Alpha'],'dropout':row['Dropout'],'parts':{"att", "ln", "time", "ffn"} }
                    elif row['Mode'] == 'bone': # test implement
                        CONFIG[f'{str(i)}']={'mode':'bone','quant':quant,'rank':row['Rank'],'parts':{"att", "ln", "time", "ffn"} }
                    elif row['Mode'] == 'pissa':
                        CONFIG[f'{str(i)}']={'mode':'pissa','quant':quant,'rank':row['Rank'],'alpha':row['Alpha'],'dropout':row['Dropout'],'parts':{"att", "ln", "time", "ffn"} }
            if Found == False:
                raise "Layer Profile Data is Invalid. Please check. orz."
        
        for row in self.array_data:
            if row['Layer'] == 'emb':
                if row['Mode'] == 'full' or row['Mode'] == 'freeze':
                    CONFIG[f'emb']={'mode':row['Mode']}
                elif row['Mode'] == 'lora':
                        CONFIG[f'emb']={'mode':'lora','rank':row['Rank'],'alpha':row['Alpha'],'dropout':row['Dropout'],'parts':{"att", "ln", "time", "ffn"} }
                
            if row['Layer'] == 'head':
                if row['Mode'] == 'full' or row['Mode'] == 'freeze':
                    CONFIG[f'head']={'mode':row['Mode']}
                elif row['Mode'] == 'lora':
                        CONFIG[f'head']={'mode':'lora','rank':row['Rank'],'alpha':row['Alpha'],'dropout':row['Dropout'],'parts':{"att", "ln", "time", "ffn"} }
                elif row['Mode'] == 'bone':
                        CONFIG[f'head']={'mode':'bone','rank':row['Rank'],'parts':{"att", "ln", "time", "ffn"} }


        return CONFIG
                




    
    #def get_mode_list(self,mode):
    #    for row in self.array_data:
    #        if row['']

