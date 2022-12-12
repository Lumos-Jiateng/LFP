from typing import List 
import marisa_trie 
from functools import partial 
from transformers import AutoTokenizer, AutoModel 
from transformers import BartTokenizer,BartForConditionalGeneration
from coin180_custom import step_list_180_with_dot
import torch

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

'''
# construct marisa trie to store choices
choices = [] # TODO: fill in the steps 
paths = []
for choice in choices:
    encoded_sent = tokenizer.encode(choice)    # for t5 must add a '2' in the begining of each choice
    paths.append(encoded_sent)
trie = MarisaTrie(paths)
'''

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
        # too much print here cancel it.
        #print("results == []")
        #print("sent:",sent)
        results = [END_TOKEN]

    if(START_TOKEN in results):
        print("why is start token in results???")

    return results

'''
# TODO: plug in your BART model inputs 
outputs = Generator.generate(
    input_ids, 
    attention_mask=attention_mask,
    max_length=max_length,
    num_beams=1,
    num_return_sequences=1,
    prefix_allowed_tokens_fn = partial(cbs, trie=trie, paths=paths),
)
'''

if __name__ == '__main__':

    #pretrained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
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
    
    inputs = pretrained_tokenizer.encode([''])
    print(inputs)
    print(cbs(batch_id=0,sent=torch.tensor([2,0,46327,5,223,27128,4,2716,5,10267]),trie=trie,paths=paths))
    '''
    outputs = pretrained_model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        #max_length=target_labels['input_ids'],
        num_beams=1,
        num_return_sequences=1,
        prefix_allowed_tokens_fn = partial(cbs, trie=trie, paths=paths),
    )

    outputs_ori = pretrained_model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        num_beams=1,
        num_return_sequences=1,
    )

    print(outputs)
    print(outputs_ori)
    print([pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs][0])
    print([pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs_ori][0])
    '''