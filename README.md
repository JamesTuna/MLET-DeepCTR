# Multi-Layer Embedding Trianing (MLET) for DeepCTR-Torch

### 1. Pytorch version 1.4.0 recommended 
- may encounter issues with regularizer in DeepCTR-Torch with higher version of pytorch

### 2. Install [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
- test are done with DeepCTR-Torch v0.2.1, compatibility with higher version not guaranteed

### 3. Data preparation
- download Avazu dataset from [here](https://www.kaggle.com/c/avazu-ctr-prediction/data)
- encode sparse features into integer values using ```data prep/preprocess_avazu.py```
- preprocessed Avazu dataset is stored as a .pkl file (can be loaded through pickle module)

### 3. Replace source codes to enable MLET for DeepCTR models 
- here we take autoInt for example, MLET implementation for other models are done in the same way
- we provide modified source code under ```utils``` and ```models``` (for DCN, NFM, AutoInt, xDeepFM, DeepFM)
- check where DeepCTR-Torch is installed ```DEEPCTR_PATH=/home/jamestuna/env/lib/python3.8/site-packages/deepctr_torch/```
- use the following commands to replace source codes:
- ```cp ./models/MLETautoInt.py ${DEEPCTR_PATH}/models/autoInt.py```
- ```cp ./utils/MLETbasemodel.py ${DEEPCTR_PATH}/models/basemodel.py```
- ```cp ./utils/MLETinputs.py ${DEEPCTR_PATH}/inputs.py```
- ```cp ./utils/MLETcore.py ${DEEPCTR_PATH}/layers/core.py```

### 4. MLET training with different inner dimension
- example code of training dcn with MLET: ```train/MLET_dcn.sh```
- this bash scripts assumes correct dataset path (.pkl file) is provided to ```train/train_dcn.py```

