python rlhf_dpo_generate_save.py  --load_model 'myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth' \
 --input_csv 'example/DPO/output_csv/rlhf_example_dataset_withreject.csv' \
 --output_save 'example/DPO/output_save/rlhf_example_dataset.save' \
 --target_pair_count 60
