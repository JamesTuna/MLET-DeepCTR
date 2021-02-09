# Multi-Layer Embedding Trianing (MLET) for DeepCTR-Torch

### 1. pytorch version 1.4.0 recommended 
- may encounter issues with regularizer in DeepCTR-Torch with higher version of pytorch

### 2. install [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
- test are done with DeepCTR-Torch v0.2.1, compatibility with higher version not guaranteed

### 3. data preparation
- download Avazu dataset from [here](https://www.kaggle.com/c/avazu-ctr-prediction/data)
- encode sparse features into integer values using ```data prep/preprocess_avazu.py```
- preprocessed Avazu dataset is stored as a .pkl file (can be loaded through pickle module)

### 3. replace source codes to enable MLET for any model that is implemented in DeepCTR-Torch 
- here we take autoInt for example, MLET implementation for other models are done in the same way
- we provide modified source code under ```utils``` and ```models``` (for DCN, NFM, AutoInt, xDeepFM, DeepFM)
- assume DeepCTR-Torch package is installed on path ```/home/jamestuna/env/lib/python3.8/site-packages/deepctr_torch/```
- use the following commands to replace source codes:
- ```cp ./models/MLETautoInt.py /home/jamestuna/env/lib/python3.8/site-packages/deepctr_torch/models/autoInt.py```
- ```cp ./utils/MLETbasemodel.py /home/jamestuna/env/lib/python3.8/site-packages/deepctr_torch/models/basemodel.py```
- ```cp ./utils/MLETinputs.py /home/jamestuna/env/lib/python3.8/site-packages/deepctr_torch/inputs.py```
- ```cp ./utils/MLETcore.py /home/jamestuna/env/lib/python3.8/site-packages/deepctr_torch/layers/core.py```

### 4. MLET training with different inner dimension
- example code of training dcn with MLET: ```train/MLET_dcn.sh```
- this bash scripts assumes correct dataset path (.pkl file) is provided to ```train/train_dcn.py```

