#2024 OpenMOSE
import csv

v7_additional_parameters = ['x_r','x_w','x_k','x_v','x_a','x_g','att.w1','att.w2','att.w0','att.a1','att.a2','att.a0','att.v1','att.v2','att.v0','att.g1','att.g2','att.k_k','att.k_a','att.r_k']

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
                      'Weight_lr_init':row['Weight_lr_init'],
                      'Weight_lr_final':row['Weight_lr_final'],
                      'State_lr_init':row['State_lr_init'],   
                      'State_lr_final':row['State_lr_final'],   
                      'Weight_decay':row['Weight_decay'],
                      'RejectParts':row['RejectParts'].split(','),              
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
                        CONFIG[f'{str(i)}']={'mode':'full','lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'lr_init_state':row['State_lr_init'],'lr_final_state':row['State_lr_final'],'RejectParts':row['RejectParts'].split(','),'weight_decay':row['Weight_decay']}
                    elif row['Mode'] == 'freeze':
                        CONFIG[f'{str(i)}']={'mode':'freeze','quant':quant,'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'lr_init_state':row['State_lr_init'],'lr_final_state':row['State_lr_final'],'RejectParts':row['RejectParts'].split(','),'weight_decay':row['Weight_decay']}
                    elif row['Mode'] == 'lora':
                        CONFIG[f'{str(i)}']={'mode':'lora','quant':quant,'rank':row['Rank'],'alpha':row['Alpha'],'dropout':row['Dropout'],'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'lr_init_state':row['State_lr_init'],'lr_final_state':row['State_lr_final'],'RejectParts':row['RejectParts'].split(','),'weight_decay':row['Weight_decay']}
                    elif row['Mode'] == 'bone': # test implement
                        CONFIG[f'{str(i)}']={'mode':'bone','quant':quant,'rank':row['Rank'],'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'lr_init_state':row['State_lr_init'],'lr_final_state':row['State_lr_final'],'RejectParts':row['RejectParts'].split(','),'weight_decay':row['Weight_decay']}
                    elif row['Mode'] == 'pissa':
                        CONFIG[f'{str(i)}']={'mode':'pissa','quant':quant,'rank':row['Rank'],'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'alpha':row['Alpha'],'dropout':row['Dropout'],'lr_init_state':row['State_lr_init'],'lr_final_state':row['State_lr_final'],'RejectParts':row['RejectParts'].split(','),'weight_decay':row['Weight_decay']}
            if Found == False:
                raise "Layer Profile Data is Invalid. Please check. orz."
        
        for row in self.array_data:
            if row['Layer'] == 'emb':
                if row['Mode'] == 'full' or row['Mode'] == 'freeze':
                    CONFIG[f'emb']={'mode':row['Mode'],'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'weight_decay':row['Weight_decay']}
                elif row['Mode'] == 'lora':
                        CONFIG[f'emb']={'mode':'lora','rank':row['Rank'],'alpha':row['Alpha'],'dropout':row['Dropout'],'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'] }
                
            if row['Layer'] == 'head':
                if row['Mode'] == 'full':
                    CONFIG[f'head']={'mode':row['Mode'],'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'lr_init_state':row['State_lr_init'],'lr_final_state':row['State_lr_final'],'weight_decay':row['Weight_decay']}
                elif row['Mode'] == 'freeze':
                    CONFIG[f'head']={'mode':row['Mode'] , 'quant':quant ,'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'weight_decay':row['Weight_decay'] } 
                elif row['Mode'] == 'lora':
                        CONFIG[f'head']={'mode':'lora','rank':row['Rank'],'alpha':row['Alpha'], 'quant':quant ,'dropout':row['Dropout'],'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'weight_decay':row['Weight_decay']}
                elif row['Mode'] == 'pissa':
                        CONFIG[f'head']={'mode':'pissa','rank':row['Rank'],'alpha':row['Alpha'], 'quant':quant ,'dropout':row['Dropout'],'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'weight_decay':row['Weight_decay']}
                elif row['Mode'] == 'bone':
                        CONFIG[f'head']={'mode':'bone','rank':row['Rank'], 'quant':quant ,'lr_init':row['Weight_lr_init'],'lr_final':row['Weight_lr_final'],'weight_decay':row['Weight_decay']}


        return CONFIG
                




    
    #def get_mode_list(self,mode):
    #    for row in self.array_data:
    #        if row['']

