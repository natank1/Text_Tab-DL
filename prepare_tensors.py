import yaml
import enum_collection as enum_obj

from transformers import AutoTokenizer,FNetTokenizer
import pickle
import torch

def get_tokenizer(data_yaml,bert_tokeinizer_name):
    if data_yaml == enum_obj.huggins_embedding.fnet.value:
        return FNetTokenizer.from_pretrained('google/fnet-base')
    tokenizer = AutoTokenizer.from_pretrained(bert_tokeinizer_name, use_fast=True)
    return tokenizer

def get_file (file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data

def bert_gen_tensor(input_t, tab_tensor, lab_all_Place, file_name,batch_enc,index_0):
    input_m = torch.squeeze(torch.index_select(batch_enc["attention_mask"], dim=0, index=torch.tensor(index_0)))
    torch.save([input_t, input_m, tab_tensor, lab_all_Place], file_name)
    return

def fnet_gen_tensor(input_t, tab_tensor, lab_all_Place, file_name,batch_enc=None,index_0=-1):
     torch.save([input_t, tab_tensor, lab_all_Place], file_name)
     return


def create_tensor_doc (feat_dic,data_yam_tok):
    batch_encoding = tokenizer.batch_encode_plus([i[0] for i in feat_dic], **data_yam_tok, return_tensors='pt')
    return batch_encoding

def generate_data_folder_w_tensor(data_lab_and_docs,data_yaml):
    batch_encoding  =  create_tensor_doc(data_lab_and_docs,data_yaml['tokenizer_dic']['params'])
    suff_val= ".pt"
    pref_val =data_yaml['tensors_folder']+"tensor_data_"
    dic_for_pic= {}

    if data_yaml['embed_val'] == enum_obj.huggins_embedding.fnet.value:
        proc_func = fnet_gen_tensor
    else:
        proc_func = bert_gen_tensor

    for i, data in enumerate(data_lab_and_docs):
            file_name = pref_val + "_" + str(i) + "_" + suff_val
            tab_tensor = torch.tensor(data[1], dtype=torch.float32)

            input_t =  torch.squeeze(torch.index_select(batch_encoding["input_ids"],dim=0,index=torch.tensor(i)))

            proc_func(input_t,tab_tensor,data[2],file_name,batch_enc=batch_encoding,index_0=i)
            dic_for_pic[file_name]= data[2]

    with   open(data_yaml['labales_file'], 'wb') as f:
         pickle.dump(dic_for_pic, f, pickle.HIGHEST_PROTOCOL)


if __name__ =='__main__':

    with open("project_yaml.yaml", 'r') as stream:
        data_yaml = yaml.safe_load(stream)
    print (data_yaml)
    tokenizer =get_tokenizer(data_yaml['embed_val'],data_yaml['bert_tokenizer_name'])





    data_lab_and_docs = get_file(data_yaml['raw_data'])

    generate_data_folder_w_tensor(data_lab_and_docs,data_yaml)
    print ("Tensors created well")



