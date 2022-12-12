from binascii import b2a_uu
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch.nn as nn
class text_baseline(nn.Module):
    def __init__(self,                 
                 config = None,
                 pretrianed_model='BartForConditionalGerneration'    
                 ):
        super().__init__()
        
        if pretrianed_model == 'BartForConditionalGerneration':
            self.pretrained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            self.pretrained_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            print("successfully init model")
        

    def forward(self, b_Prompt,b_Predict,device):
        #print("model usage:")
        batch_size = len(b_Predict)
        #print(b_Prompt)
        inputs = self.pretrained_tokenizer(b_Prompt, truncation=True, padding=True,return_tensors='pt')
        target_labels = self.pretrained_tokenizer(b_Predict, truncation=True, padding=True,return_tensors='pt')
        Seq2seq_output=self.pretrained_model(input_ids = inputs['input_ids'].to(device,non_blocking=True),
                                            attention_mask = inputs['attention_mask'].to(device,non_blocking=True),
                                            labels = target_labels['input_ids'].to(device,non_blocking=True))

        generation_loss=Seq2seq_output[0]

        return generation_loss
        