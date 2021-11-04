import numpy as np

import pickle
from pytorch_tabnet.pretraining import TabNetPretrainer

def get_file (tab_file_name):
    with open(tab_file_name, 'rb') as f:
        data = pickle.load(f)
        return data
if __name__ =='__main__':
    pick_name = "/code_pickle/pickl_data.p"
    embed_name = "/code_pickle/embed_data.p"

    data =get_file(pick_name)
    X_data =[i[1] for i in data]

    X_train =np.asarray(X_data)

    clf = TabNetPretrainer()  # TabNetRegressor()
    clf.fit(
        X_train[:100])


    print ("prepare the data")
    t1 =clf.predict(X_train)

    tt =clf.predict([i  for i in X_train])[1]
    data1 =[(i[0],j,i[2]) for i,j in zip(data,tt)]

    with   open(embed_name, 'wb') as f:
        pickle.dump(data1, f, pickle.HIGHEST_PROTOCOL)
