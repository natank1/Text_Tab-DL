import yaml
import os
from sklearn.model_selection import train_test_split
from dataet_tensor_training import dataset_tensor
import torch
import torch.nn  as nn
from torch.utils.data import DataLoader
from loss_and_re_manager import loss_manager
from my_model import my_model
from tqdm import  tqdm
import numpy as np

def convert_eval_score_and_label_to_np(y_label, y_scores):
    y_true0 = [i.cpu().detach().numpy() for i in y_label]
    y_true = np.concatenate(y_true0)
    y_pred0 = [i.cpu().detach().numpy() for i in y_scores]
    y_pred = np.concatenate(y_pred0)
    return y_true, y_pred


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print ("go training")
    with open("project_yaml.yaml", 'r') as stream:
        data_yaml = yaml.safe_load(stream)
    list_of_files =os.listdir(data_yaml['tensors_folder'])
    list_of_files =[data_yaml['tensors_folder']+i for i in list_of_files]
    X_train,X_test =train_test_split(list_of_files,test_size=0.2)
    print (len(X_train),len(X_test))
    train_t =dataset_tensor(X_train,data_yaml['embed_val'])
    train_loader = DataLoader(train_t, batch_size=data_yaml['batch_size'], shuffle=True)

    test_t = dataset_tensor(X_test, data_yaml['embed_val'])
    test_loader = DataLoader(test_t, batch_size=data_yaml['batch_size'], shuffle=True)

    #For GPU usage
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    model = my_model(data_yaml)
    if data_yaml['improv_model']:
        print("Loading mode")
        model_place = data_yaml['pre_trained_folder']
        print (model_place)
        model.load_state_dict(torch.load(model_place, map_location='cpu'))

    if device:
        model =model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-5,  # This is the value Michael used.
                                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                  )
    loss_man = loss_manager(data_yaml['batch_size'], data_yaml['target_val'],data_yaml['reg_term'])
    model.train()
    for i_epoc in range(data_yaml['n_epochs']):
        print ("i_epoch=",i_epoc)

        running_loss = 0.0
        counter_a=0
        for batch_idx, data in enumerate(train_loader):

            a,b,d,c=  data
            if device :
                a=a.to(device)
                b=b.to(device)
                c=c.to(device)
                d=d.to(device)

            ss=model(x=d, input_ids=a,attention_mask=b)


            loss =loss_man.crit(ss,c)
            running_loss += loss.item()

            loss.backward()
            print(loss,batch_idx)
            optimizer.step()
            optimizer.zero_grad()


        print ("Epoch loss= ",running_loss/(counter_a+0.))
        print ("End of epoc")
        torch.save(model.state_dict(),  data_yaml['models_folder'] + "model_epoch_" + str(i_epoc) + ".bin")
    print.info ("Training worked well")

    print ("Now eval ")
    with torch.no_grad():
        with tqdm(total=len(test_loader), ncols=70) as pbar:
            labels = []
            predic_y = []
            for batch_idx, data in enumerate(test_loader):

                a, b, d, c = data
                if device:
                    a = a.to(device)
                    b = b.to(device)
                    c = c.to(device)
                    d = d.to(device)
                labels.append(c)
                outp = model(x=d, input_ids=a, attention_mask=b)
                probs = nn.functional.softmax(outp, dim=1)
                predic_y.append(probs)
                pbar.update(1)

            y_true, y_pred = convert_eval_score_and_label_to_np(labels, predic_y)
    print ("Eval is over. from here everyone can plot his grpahs ")