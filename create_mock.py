import random
import pickle
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups,fetch_covtype



def get_mock_db(my_pickle_name):
    covert=fetch_covtype(data_home=None, download_if_missing=True, random_state=None, shuffle=False)

    categories = ['alt.atheism', 'soc.religion.christian',           'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',   remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)
    clean_text =[  (i.replace('\n','').replace('\t','').replace('\\','') ,j) for i,j in zip(twenty_train.data,twenty_train.target)]
    clean_text =[  i for i in clean_text if len(i[0])>20]
    len_0 =len([i for i in clean_text if i[1]==0])
    len_1 =len([i for i in clean_text if i[1]==1])
    len_2= len([i for i in clean_text if i[1]==2])
    len_3 =len([i for i in clean_text if i[1]==3])
    print (len_0,len_1,len_2,len_3)
    a,b =covert.data.shape
    cov_0 = [covert.data[j] for j in range(a) if covert['target'][j]==4 ][:len_0]
    cntr=0
    for j in range(len(clean_text)):
        if clean_text[j][1]==0:
            a1,b1 =clean_text[j]
            clean_text[j]= (a1,cov_0[cntr],b1)
            cntr+=1
        if cntr ==len_0:
            break

    cov_1 = [covert.data[j] for j in range(a) if covert['target'][j]==1 ][:len_1]
    cntr=0
    for j in range(len(clean_text)):
        if (len(clean_text[j])==2) and clean_text[j][1]==1:
            a1,b1 =clean_text[j]
            clean_text[j]= (a1,cov_1[cntr],b1)
            cntr+=1
        if cntr ==len_1:
            break

    cov_2 = [covert.data[j] for j in range(a) if covert['target'][j]==3 ][:len_2]
    cntr=0
    for j in range(len(clean_text)):
        if  (len(clean_text[j])==2) and clean_text[j][1]==2:
            a1,b1 =clean_text[j]
            clean_text[j]= (a1,cov_2[cntr],b1)
            cntr+=1
        if cntr ==len_2:
            break
    cov_3 = [covert.data[j] for j in range(a) if covert['target'][j]==3 ][:len_3]
    cntr=0
    for j in range(len(clean_text)):
        if (len(clean_text[j])==2) and clean_text[j][1]==3:
            a1,b1 =clean_text[j]
            clean_text[j]= (a1,cov_3[cntr],b1)
            cntr+=1
        if cntr ==len_3:
            break

    with   open(my_pickle_name, 'wb') as f:
        pickle.dump(clean_text, f, pickle.HIGHEST_PROTOCOL)
    print ("files_genrated")
    return
if __name__ =='__main__':
    pick_name ="/Users/natankatz/PycharmProjects/pythonProject/pythonProject1/iiris_huggings/code_pickle/pickl_data.p"
    get_mock_db(pick_name)
