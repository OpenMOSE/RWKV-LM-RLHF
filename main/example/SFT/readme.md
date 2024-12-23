## SFT Fine-tuning

Infinite-Context SFT with SmoothLoss

Example Training Steps with x060 7B Model

Notice.

Due to the poor quality of my code, NaN occurs very rarely. It works after a few tries.

This is best(for me) for multi-turn-instruct tuning.

### Step.-1 Download Model
  - make directory, myfolder/models
  - download RWKV-x060-Jpn-7B-20240816-ctx4096.pth in HF
  - copy to myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth

### Step.0 Prepare Dataset
  - Prepare CSV file with (prompt,chosen keys)
  - if not contain reject, you can use this script for generating reject
    ```
    ./example/SFT/step-0-csvtojsonl.sh
    ```
### Step.1 Generate Tokenized File
  - ```
    ./example/SFT/step-1-jsonltoh5.sh
    ```
### Step.2 Train
  - ```
    ./example/SFT/step-2-train-sft.sh
    ```

if you have any questions,pm me.

OpenMOSE