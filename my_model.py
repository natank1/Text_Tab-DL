import torch
import torch.nn as nn
from transformers import BertPreTrainedModel,BertModel,BertConfig,FNetModel,FNetConfig
from transformers import  DistilBertForSequenceClassification,BertForSequenceClassification,FNetForSequenceClassification
from transformers.models.distilbert import DistilBertModel
import enum_collection as enum_obj

class my_fnet(FNetForSequenceClassification):

    def __init__(self,config,dim=768):
        super(my_fnet, self).__init__(config )
        self.dim= dim
        self.num_labels = 4
        self.distilbert = FNetModel(config)
        self.init_weights()

        self.pre_classifier = nn.Linear(self.dim, self.dim)
    def forward(self,  input_ids=None):
            outputs = self.distilbert( input_ids=input_ids)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            return pooled_output




class my_bert(BertForSequenceClassification):
    def __init__(self,config,dim=768):
        super(my_bert, self).__init__(config )
        self.dim= dim
        self.num_labels = 4
        self.distilbert = BertModel(config)
        self.init_weights()

        self.pre_classifier = nn.Linear(self.dim, self.dim)
    def forward(self,
                    input_ids=None,
                    attention_mask=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,

                    ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            distilbert_output = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            return pooled_output



class my_dist(DistilBertForSequenceClassification):
   #Taking from transformers/models/distilbert/modeling_distilbert.py
    def __init__(self,config,dim=768):
        super(my_dist, self).__init__(config )
        self.dim= dim
        print ("natan model")
        self.num_labels = 4
        self.config = config
        self.distilbert = DistilBertModel(config)
        self.init_weights()
        print ("dist")
        self.pre_classifier = nn.Linear(self.dim, self.dim)
    def forward(self,
                    input_ids=None,
                    attention_mask=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,

                    ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            distilbert_output = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            return pooled_output

class my_model(nn.Module):
    def __init__(self,data_yaml,my_tab_form=1,dim=768):
        super(my_model, self).__init__( )
        self.forward =self.bert_forward
        if data_yaml['embed_val'] == enum_obj.huggins_embedding.distil_bert.value:
            self.dist_model  = my_dist.from_pretrained('distilbert-base-multilingual-cased',num_labels=4 )
        elif data_yaml['embed_val'] == enum_obj.huggins_embedding.base_bert.value:
            self.dist_model = my_bert.from_pretrained('bert-base-multilingual-cased', num_labels=4)
        else:
            self.dist_model = my_fnet.from_pretrained('google/fnet-base', num_labels=4)
            self.forward = self.fnet_forward
        if my_tab_form>0 :
           localtab= data_yaml['tab_format']
        else :
            localtab =my_tab_form
        if localtab ==  enum_obj.tab_label.no_tab.value:
            print ("no_tab")
            self.embed_to_fc = self.cat_no_tab
            self.tab_dim = 0
        else :
            self.embed_to_fc = self.cat_w_tab
            self.tab_dim =data_yaml['tab_dim']
        print ("natan model")
        self.dim=dim
        self.num_labels =4

        self.pre_classifier = nn.Linear( self.dim, self.dim)
        self.inter_m0= nn.Linear(self.dim +self.tab_dim,216)

        self.inter_m1 = nn.Linear(216,64)
        self.inter_m1a = nn.Linear(64, 32)


        self.inter_m3 = nn.Linear(32, self.num_labels)
        self.classifier = nn.Linear(self.dim, self.num_labels)
        self.dropout = nn.Dropout(0.2)

        # self.init_weights()

    def cat_no_tab (self,hidden,x):
        return hidden
    def cat_w_tab (self,hidden,x):
        return torch.cat((hidden,x),dim=1)

    def fnet_forward(self, x,
                 input_ids=None,attention_mask=None):

        hidden_state = self.dist_model(input_ids)

        pooled_output = torch.cat((hidden_state, x), dim=1)
        pooled_output = self.odebl.forward(pooled_output)

        pooled_output = self.inter_m0(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        pooled_output = self.inter_m1(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        pooled_output = self.inter_m1a(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.inter_m3(pooled_output)  # (bs, num_labels)
        return logits
    def bert_forward(self,x,
        input_ids=None,
        attention_mask=None) :
        hidden_state =self.dist_model(input_ids,attention_mask)

        pooled_output = self.embed_to_fc(hidden_state,x)

        pooled_output = self.odebl.forward(pooled_output)
        pooled_output = self.inter_m0(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        pooled_output = self.inter_m1(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        pooled_output = self.inter_m1a (pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.inter_m3(pooled_output)  # (bs, num_labels)
        return logits


if __name__ =='__main__':
    xx = torch.randint(20, (8, 512))
    x2 =torch.randint(2,(8,512))
    yy = torch.randn((8, 108))
    m2= my_model()


    tt =  m2(x=yy,input_ids=xx)
    print (tt.shape,"ghghhg")
