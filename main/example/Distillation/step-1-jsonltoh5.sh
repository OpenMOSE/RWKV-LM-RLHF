python distillation_generate_h5.py --input_folder 'example/Distillation/input_jsonl' \
 --load_initial_state '' \
 --output_parquet 'example/Distillation/output_h5/distillation_dataset.h5' \
 --load_model 'myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth' \
 --strategy 'cuda fp16'
