from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import Tensor
import numpy as np
import math


class IntimacyEstimator:
    def __init__(self, cuda = False):

        model_path = 'pedropei/question-intimacy'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, num_labels=1, output_attentions=False,
                                                     output_hidden_states=False, cache_dir = './model_cache')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1,
                                                                 output_attentions=False, output_hidden_states=False,cache_dir = './model_cache')
        self.cuda = cuda
        if cuda:
            self.model.cuda()

    def data_iterator(self, train_x, batch_size):
        n_batches = math.ceil(len(train_x) / batch_size)
        for idx in range(n_batches):
            x = train_x[idx * batch_size:(idx + 1) * batch_size]
            yield x

    def padding(self, text, pad, max_len=50):
        return text if len(text) >= max_len else (text + [pad] * (max_len - len(text)))

    def encode_batch(self, text):

        tokenizer = self.tokenizer
        t1 = []
        for line in text:
            t1.append(self.padding(tokenizer.encode(line, add_special_tokens=True, max_length=50, truncation=True),
                              tokenizer.pad_token_id))

        return t1

    def predict(self, text, type = 'single', batch_size=128, tqdm=None):
        if type == 'single':
            ids = [self.padding(self.tokenizer.encode(text,add_special_tokens = True, max_length = 50,truncation=True),self.tokenizer.pad_token_id)]
            if self.cuda:
                ids = Tensor(ids).cuda().long()
            else:
                ids = Tensor(ids).long()
            outputs = self.model(ids)
            return outputs[0].cpu().data.numpy()[0][0]

        elif type == 'list':
            ids = self.encode_batch(text)
            with torch.no_grad():
                if self.cuda:
                    input_ids = Tensor(ids).cuda().long()
                else:
                    input_ids = Tensor(ids).long()
                outputs = self.model(input_ids)
            return list(outputs[0].cpu().data.numpy().flatten())

        elif type == 'long_list':
            test_iterator = self.data_iterator(text, batch_size)
            all_preds = []
            all_res = []
            if tqdm:
                #print('Please use tqdm to track the progress of model predictions')
                #return None
                for x in tqdm(test_iterator, total=int(len(text) / batch_size)):

                    ids = self.encode_batch(x)

                    with torch.no_grad():
                        if self.cuda:
                            input_ids = Tensor(ids).cuda().long()
                        else:
                            input_ids = Tensor(ids).long()
                        outputs = self.model(input_ids)

                    predicted = outputs[0].cpu().data.numpy()
                    all_preds.extend(predicted)

                all_res = np.array(all_preds).flatten()
                return list(all_res)
            else:
                for x in test_iterator:

                    ids = self.encode_batch(x)

                    with torch.no_grad():
                        if self.cuda:
                            input_ids = Tensor(ids).cuda().long()
                        else:
                            input_ids = Tensor(ids).long()
                        outputs = self.model(input_ids)

                    predicted = outputs[0].cpu().data.numpy()
                    all_preds.extend(predicted)

                all_res = np.array(all_preds).flatten()
                return list(all_res)

        else:
            print('Wrong input type, please use \'single\',\'list\' or \'long_list\'')