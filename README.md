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

Remember to change ```YOUR_JSON_FILE YOUR_TARGET_IMAGE_DIRECTORY YOUR_IMAGE_ROOT_PATH ``` into your ideal file path when running the preprocessing code.

To obtain the whole COIN dataset, visit [Official Web Page of COIN](https://coin-dataset.github.io/) to download the videos / annotation file.

The overall file structure should be as below after the preprocessing:

    coin_preprocessing
    ---Json_for_LM
       ---coin_train_step_3.json
       ---coin_val_step_3.json
       ---coin_test_step_3.json
       ---coin_train_step_4.json
       ...
    ---COIN_Images
       ---Train_Image_all
          ---0
             ----8NaVGEccgc (youtube_id)
                 ---image_m_0.jpg
                 ---image_m_1.jpg
                 ...
             ...
          ---1
             ...
       ---Val_Image_all
          ...
       ---Test_Image_all
          ...
    ---Json_for_DR
       ---Multi-Image
          ---one_train.json
          ---one_val.json
          ---one_test.json
          ---two_train.json
          ...
       ---Single-Image
          ...

### Train / Evaluate the Language Model Module

We used the Hugging Face transformers library for our language model implementation, check the documents on [BART - documentation](https://huggingface.co/transformers/v4.9.0/model_doc/bart.html)

To train / evaluate the language Model, run:
    
    cd LFP
    cd Language_pred
    python lm_predict.py --config YOUR_CONFIG --output_dir YOUR_OUTPUT_DIR --result_path YOUR_RESULT_PATH --pred_path YOUR_PRED_PATH --prompt_mode PROMPT
    
 ```YOUR_RESULT_PATH``` and ```YOUR_PRED_PATH``` will record the accuracy number of the language prediction and the language predict result of all test samples.
 
 ```PROPMT``` mode allows you to change the language model prompt style according to our paper.
    
### Train / Evaluate the Double Retrieval Module

We use the [Pretrained BLIP Model checkpoint](https://github.com/salesforce/BLIP) to initialize our model. Download the checkpoint [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth)

To train / evaluate the Double Retrieval Model, run:
    
    cd LFP
    cd Double_retrieval
    python coin_retrieve.py --config YOUR_CONFIG --output_dir YOUR_OUTPUT_DIR --result_path YOUR_RESULT_PATH --write_path YOUR_WRITE_PATH 
    
### Train / Evaluate the Modularized FrameWork

After Obtaining the checkpoints on our LM module and DR module independently, we can evaluate the performance of our Modularized FrameWork.

To evaluate our Modularized FrameWork, simply run:
    
    cd LFP
    cd Modularized_Framework
    python coin_eval.py --config YOUR_CONFIG --output_dir YOUR_OUTPUT_DIR --result_path YOUR_RESULT_PATH --write_path YOUR_WRITE_PATH 

### Citation

### Acknowledgement 

The implementation our code repository relies on resources from BLIP, Huggingface Transformers, and timm. We thank the original authors for their open-sourcing.
