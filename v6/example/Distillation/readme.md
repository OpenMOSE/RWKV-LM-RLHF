## Distillation Fine-tuning

Infinite-Context Distillation + SFT with SmoothLoss

Example Training Steps with x060 7B Model

Notice.

Due to the poor quality of my code, NaN occurs very rarely. It works after a few tries.

This is best(for me) for distillation.

### Step.-1 Download Model
  - make directory, myfolder/models
  - download RWKV-x060-Jpn-7B-20240816-ctx4096.pth in HF
  - copy to myfolder/models/RWKV-x060-Jpn-7B-20240816-ctx4096.pth

### Step.0 Prepare Dataset
  - Prepare CSV file with (prompt,chosen keys)
  - if not contain reject, you can use this script for generating reject
    ```
    ./example/Distillation/step-0-csvtojsonl.sh
    ```
### Step.1 Generate Tokenized File
  - ```
    ./example/Distillation/step-1-jsonltoh5.sh
    ```
### Step.2 Train
  - ```
    ./example/Distillation/step-2-train-distillation.sh
    ```

if you have any questions,pm me.

OpenMOSE