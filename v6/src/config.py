LAYER_CONFIG = {
    '0':{'mode':'freeze','quant':'nf4'},
    '1':{'mode':'lora','quant':'nf4','rank':'8','alpha':'16','parts':{"att", "ln", "time", "ffn"}},
    '2':{'mode':'full'},
}

def update_layer_config(new_config):
    global LAYER_CONFIG
    LAYER_CONFIG.update(new_config)