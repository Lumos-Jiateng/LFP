from models.med import BertConfig
from models.nvlr_encoder import BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint
from data.crosstask_steps import Crosstask_steps
import numpy as np

language_sup_path = '/nfs4-p1/ljt/github/PTU/Double_retrieval/language_sup/features.npy'

class double_text_align(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 #image_size = 384,
                 #vit = 'base',
                 #vit_grad_ckpt = False,
                 #vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 #queue_size = 57600,
                 #momentum = 0.995,
                 #negative_all_rank = False,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        # can use a projection to preprocess visual features
        # self.visual_proj = nn.Linear(3200,768)

        #init text side BertModel
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = 3200  #vision_width embedding = 3200 here
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)        
        text_width = self.text_encoder.config.hidden_size
        
        # vector dimension for 'state vectors'
        self.predict_visual_dim = 512
        
        #Four heads separately for two terms of training losses
        self.cls_head_for_ff = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 105)
                )  
        self.cls_head_for_lf = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 105)
                )    
        self.language_head_for_ff = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, self.predict_visual_dim)
                ) 
        self.language_head_for_lf = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, self.predict_visual_dim)
                )     

        # project steps embeddings
        self.text_proj = nn.Linear(text_width, self.predict_visual_dim)    

        # temp parameter, following blip design
        self.temp = nn.Parameter(0.07*torch.ones([]))      


    def forward(self,ff_start,ff_end,lf_start,lf_end,ff_action_label,lf_action_label,text_description):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        # using ff_start + lf_end to predict  
        # get visual atts    

        # problem occurred : ff_start and lf_end are not bz*seq_len*dim but bz*dim -->unsqueeze
        bz = ff_start.shape[0]
        #ff_start = torch.unsqueeze(ff_start,1)
        #lf_end = torch.unsqueeze(lf_end,1)
        ff_start_atts = torch.ones(ff_start.size()[:-1],dtype=torch.long).to(ff_start.device) 
        lf_end_atts = torch.ones(lf_end.size()[:-1],dtype=torch.long).to(ff_start.device) 

        #text_string = 'The task is ' + target + ' and there are ' + text_description + ' in between'
        text = self.tokenizer(text_description, padding='max_length', return_tensors="pt").to(ff_start.device) 
        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [ff_start,lf_end],
                                   encoder_attention_mask = [ff_start_atts,lf_end_atts],        
                                   return_dict = True,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]         
        ff_prediction = self.cls_head_for_ff(hidden_state)
        lf_prediction = self.cls_head_for_lf(hidden_state)
        ff_embedding = self.language_head_for_ff(hidden_state)
        lf_embedding = self.language_head_for_lf(hidden_state)
        
        ###============== classification loss ===================###

        #change to one-hot targets
        ff_target = F.one_hot(ff_action_label, num_classes=105) #[1,3] -> [1,0,0...],[0,0,1...]
        lf_target = F.one_hot(lf_action_label, num_classes=105) 

        loss_ff = F.cross_entropy(ff_prediction, ff_action_label)
        loss_lf = F.cross_entropy(lf_prediction, lf_action_label) # do not need one hot in cross-entropy

        ###============== language contrastive loss ===================###
        step_feat = np.load(language_sup_path)
        step_feat = torch.tensor(step_feat).to(ff_start.device) # load existing feat as ground truth.
        
        ff_sim_matrix = ff_embedding @ step_feat.t() / self.temp
        lf_sim_matrix = lf_embedding @ step_feat.t() / self.temp
        language_loss_ff = -torch.sum(F.log_softmax(ff_sim_matrix, dim=1)*ff_target,dim=1).mean()
        language_loss_lf = -torch.sum(F.log_softmax(lf_sim_matrix, dim=1)*lf_target,dim=1).mean()
        
        return loss_ff , loss_lf , language_loss_ff , language_loss_lf
        