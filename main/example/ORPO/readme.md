## RLHF ORPO Fine-tuning

Odds Ratio Preference Optimization

Example Training Steps with x060 7B Model

Notice.

Currently, only Single-turn RLHF support(i think its enough)

### Step.-1 Download Model
  - make directory, myfolder/models
  - download RWKV-x060-Jpn-7B-20240816-ctx4096.pth in HF
  - copy to myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth

### Step.0 Prepare Dataset
  - Prepare CSV file with (prompt,chosen keys)
  - if not contain reject, you can use this script for generating reject
    ```
    ./example/ORPO/step-0-make-reject.sh    
    ```
### Step.1 Generate Tokenized File
  - ```
    ./example/ORPO/step-1-make-save.sh
    ```
### Step.2 Train
  - ```
    ./example/ORPO/step-2-train-orpo.sh
    ```