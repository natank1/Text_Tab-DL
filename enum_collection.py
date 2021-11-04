from enum import Enum

class huggins_embedding(Enum):
    distil_bert = 0
    base_bert = 1
    fnet = 2

class tab_label(Enum):
    no_tab =0
    raw_tab = 1
    tab_net =2

class regulation_term(Enum):
    no_reg = 0
    only_imp = 1
    both_term = 2
    wassertein =3