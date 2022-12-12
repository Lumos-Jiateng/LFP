# A-Language-First-Approach-to-Procedural-Planning
This is the official code repository for our paper: A Language First Approach to Procedural Planning.

#### Our Modularized Framework
![Our Modularized Framework: The Right part is a Double retrieval model which obtains both the start step and the end step with the input image and Textual prompt. The Left part is a finetuned language model based on ground truth steps, which is designed to predict the intermediate steps. We use the integration of these two models to do procedural planning task.](https://github.com/Lumos-Jiateng/A-Language-First-Approach-to-Procedural-Planning/blob/main/images/model_architecture_double_infer-page1.jpg)
Our Modularized Framework: The Right part is a Double retrieval model which obtains both the start step and the end step with the input image and Textual prompt. The Left part is a finetuned language model based on ground truth steps, which is designed to predict the intermediate steps. We use the integration of these two models to do procedural planning task.

### Catalog 
#### Version-1.0 Update:
  1. Data Preprocessing code on COIN dataset.
  2. Training and Testing code for our Modularized Framework on COIN.

Our code is buit on torch 1.12 and CUDA 11.3, to install the dependencies, run:
    
    pip install requirements.txt
    
### Data preprocessing
The COIN dataset is originally instructional videos, with each piece of annotation provides the start time and the end time of an action (step). In our preprocessing code, we do the following things:
  1. Reorganize the dataset and split the whole dataset into train / val / test.   ---run split_by_id.py
  2. Generate images from the original COIN videos. (Including multi-frame and single-frame).   ---run generate_images.py 
  3. Generate json file for training the language model and the double retrieval model independently.   ---run generate_json_final.py 

To obtain the whole COIN dataset, visit [Official Web Page of COIN](https://coin-dataset.github.io/)
