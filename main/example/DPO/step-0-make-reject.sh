python rlhf_generate_reject_csv.py --load_model 'myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth' \
 --input_csv 'example/DPO/input_csv/rlhf_example_dataset.csv' \
 --output_csv 'example/DPO/output_csv/rlhf_example_dataset_withreject.csv' \
 --strategy 'cuda fp16' 
