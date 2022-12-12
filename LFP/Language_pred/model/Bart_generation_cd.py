from binascii import b2a_uu
from transformers import BartTokenizer,BartConfig
from model.modeling_bart_cd import BartForConditionalGeneration
import torch.nn as nn
from model.coin180_custom import step_list_180_with_dot
from functools import partial
from typing import List 
import marisa_trie 

class MarisaTrie(object):
    def __init__(
        self,
        sequences: List[List[int]] = [],
        cache_fist_branch=True,
        max_token_id=256001,
    ):

        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        self.cache_fist_branch = cache_fist_branch
        #print(sequences)
        if self.cache_fist_branch:
            self.zero_iter = list({sequence[0] for sequence in sequences})
            assert len(self.zero_iter) == 1
            self.first_iter = list({sequence[1] for sequence in sequences})

        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) for sequence in sequences
        )

    def get(self, prefix_sequence: List[int]):
        if self.cache_fist_branch and len(prefix_sequence) == 0:
            return self.zero_iter
        elif (
            self.cache_fist_branch
            and len(prefix_sequence) == 1
            and self.zero_iter == prefix_sequence
        ):
            return self.first_iter
        else:
            key = "".join([self.int2char[i] for i in prefix_sequence])
            return list(
                {
                    self.char2int[e[len(key)]]
                    for e in self.trie.keys(key)
                    if len(e) > len(key)
                }
            )

    def __iter__(self):
        for sequence in self.trie.iterkeys():
            yield [self.char2int[e] for e in sequence]

    def __len__(self):
        return len(self.trie)

    def __getitem__(self, value):
        return self.get(value)

def cbs(batch_id, sent, trie, paths):
    '''given a input prefix sequence, this function returns all the possible next words in a list'''
    START_TOKEN = 0
    SEP_TOKEN = 4 # a SEP_TOKEN is defined here as ' . '
    END_TOKEN = 2
    # nonlocal trie, paths  # this makes trie refer to the outside variable trie

    # first convert the input tensor to list
    token_list = sent.tolist()


    # sometimes bart generates the start_token even when its not in the allowed_tokens list, here we remove them
    while START_TOKEN in token_list[2:]:
        temp = token_list[2:]
        temp.remove(START_TOKEN)
        token_list[2:] = temp


    # since the model always generates 2 as the first argument, we manually remove it every time
    if token_list == [2]:
        results = START_TOKEN
        return results
    else:
        token_list = token_list[1:]

    # if the current last token is a sep_token, then generate on a start_token or end the generation with end_token
    if token_list[-1] == SEP_TOKEN:

        # first update the tree to remove the previously generated sentence
        # if there are other sep_tokens in the sequence, take the sentence after the second to last sep_token
        '''
        if token_list.count(SEP_TOKEN) > 1:
            token_list_minus_last_SEP = token_list[:len(token_list)-1]
            p = len(token_list_minus_last_SEP) - 1 - token_list_minus_last_SEP[::-1].index(SEP_TOKEN) # this should give the position of the second to last sep_token
            sent_just_generated = [START_TOKEN] + token_list[p+1:] + [END_TOKEN]
            paths.remove(sent_just_generated)   # remove the first sentence in paths that matches sent_just_generated
            if paths != []:
                trie = MarisaTrie(paths)
            else:
                results = [END_TOKEN]

        # if there is no other sep_tokens in the sequence take everything from begining to this sep_token
        else:
            sent_just_generated = token_list + [END_TOKEN]
            paths.remove(sent_just_generated)   # remove the first sentence in paths that matches sent_just_generated
            if paths != []:
                trie = MarisaTrie(paths)
            else:
                results = [END_TOKEN]
        '''
        # now use the new tree to get results
        results = trie.get([START_TOKEN]) + [END_TOKEN]
        
    # if we are in the middle of generating the ith sentence (i>1), cut off the part from previous sentences
    elif SEP_TOKEN in token_list:
        res = len(token_list) - 1 - token_list[::-1].index(SEP_TOKEN)    # get the position of the last sep token
        prefix = [START_TOKEN] + token_list[res+1:]
        results = trie.get(prefix)

    # if we are generating the first sentence
    else:
        results = trie.get(token_list)

    # in the case where the model chose a token outside of our plans, end the generation here to save time
    if results == []:
        print("results == []")
        print("sent:",sent)
        results = [END_TOKEN]

    if(START_TOKEN in results):
        print("why is start token in results???")

    return results
'''
# construct marisa trie to store choices
choices = [] # TODO: fill in the steps 
paths = []
for choice in choices:
    encoded_sent = tokenizer.encode(choice)    # for t5 must add a '2' in the begining of each choice
    paths.append(encoded_sent)
trie = MarisaTrie(paths)
'''

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

import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    
    model = text_baseline()
    
    checkpoint_path = '/nfs4-p1/ljt/github/LFP/Language_pred/output/coin_3_predict_all_123/text_baseline/checkpoint_coin_all_123_epoch_11_mode_2_T_1.pth'
    print(f'load_checkpoint_from{checkpoint_path}')
    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt['model']
    msg = model.load_state_dict(state_dict,strict=False)
    print(msg)           
    model = model.to(device)
    #
    #"show the blank paper", "fold or fire the blank paper", "show the money to the audience"
    pretrained_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    choices =  step_list_180_with_dot
    # TODO: fill in the steps 
    paths = []
    for choice in choices:
        encoded_sent = pretrained_tokenizer.encode(choice)    # for t5 must add a '2' in the begining of each choice
        print(encoded_sent)
        paths.append(encoded_sent)
        #paths = [[0,30534,514,2],[0,30534,10580,2]]
    trie = MarisaTrie(paths)
    
    # define the new function of computing available decode ids
    #prefix_allowed_tokens_fn = partial(cbs, trie=trie, paths=paths)
    
    Test = ['For Task PerformPaperToMoneyTrick Given the first step and the last step, predict the intermediate one step.show the blank paper.show the money to the audience.']
    Inputs=model.pretrained_tokenizer(Test, truncation=True, padding=True,return_tensors='pt')
        
    #beam_search used in generation
    summary_ids = model.pretrained_model.generate(Inputs['input_ids'].to(device,non_blocking=True), 
                                                  num_beams=8, max_length=50, early_stopping=True,
                                                  prefix_allowed_tokens_fn = partial(cbs, trie=trie, paths=paths),)
    
    print(summary_ids)
    print([model.pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0])
    