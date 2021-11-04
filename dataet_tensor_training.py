import torch
import enum_collection as enum_obj
import random

class dataset_tensor :
    def __init__ (self, list_of_files, embed_val ):

        self.list_of_files =list_of_files
        random.shuffle(list_of_files)
        if embed_val==enum_obj.huggins_embedding.fnet.value:
            self.ref =[0,1,2]
        else:
            self.ref=[1,2,3]

    def __getitem__(self,idx):
       aa =torch.load(self.list_of_files[idx])
       return aa[0], aa[self.ref[0]], aa[self.ref[1]],aa[self.ref[2]]


    def __len__(self) :
        return len(self.list_of_files)

