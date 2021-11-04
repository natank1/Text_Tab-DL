import torch
import torch.nn as nn


from  torch.nn import CrossEntropyLoss
ay1=-2.5
ay2=0.4
ax1=0.1
ax2=1
mu =(ay2-ay1)/(ax2-ax1)
beq= ay1-mu*ax1

by1=-2.5
by2=0.4
bx1=0.4
bx2=0.
bmu =(by2-by1)/(bx2-bx1)
beq2= by1-bmu*bx1
#Wasserstein settings
wy1=-2.5
wy2=2.5
wx1=9e-6
wx2=1e-5
wmu =(wy2-wy1)/(wx2-wx1)
wbeq= wy1-wmu*wx1

# print (mu,beq)
# y1
# y2
# x1
# x2
class loss_manager:
    def __init__(self,batch_size,succ_val,reg_term):
        self.crit_pr=CrossEntropyLoss()
        self.succ_val =succ_val
        self.len_b =batch_size
        self.mu =torch.tensor(mu).type(torch.float32)
        self.beq = torch.tensor(beq).type(torch.float32)
        self.bmu = torch.tensor(bmu).type(torch.float32)
        self.beq2 = torch.tensor(beq2).type(torch.float32)
        self.wmu =wmu
        self.wbeq=wbeq
        self.rep_batch =torch.reciprocal(torch.tensor(self.len_b )).type(torch.float32)
        if reg_term ==0 :
            self.reg_func =self.no_reg
        elif reg_term ==1:
            self.reg_func =self.calc_only_reg
        elif reg_term == 2:
            self.reg_func = self.regulation_term
        else:
            self.reg_func= self.wasserstein_term

    def no_reg(self,predictions, labels):
        return 0.0

    def calc_only_core (self,predictions, labels):
        condition_t = torch.where(labels == self.succ_val, 0, 1)

        n2 = torch.sum(condition_t)
        probs = nn.functional.softmax(predictions, dim=1)[:, self.succ_val]
        tan_prob = 100. * (1 + torch.tanh(self.mu * probs + self.beq))
        score1 = torch.dot(tan_prob, condition_t.type(torch.float32))
        return score1, n2, condition_t, probs

    def calc_only_reg(self,predictions, labels):
        score, n2 ,_,_ = self.calc_only_core( predictions, labels)
        return score/n2

    def regulation_term (self,predictions, labels):

        score1, n2 ,condition_t,probs =self.calc_only_core(predictions, labels)
        zero_cn = self.len_b - n2
        tan_prob2 = 100.*(1+torch.tanh(self.bmu*probs+self.beq2))
        score2 = torch.dot(tan_prob2, (1-condition_t).type(torch.float32))
        score =(zero_cn*score2 +n2*score1)/self.len_b
        return score
    def crit (self,predictions, labels):
        mm=self.reg_func(predictions, labels)
        return self.crit_pr(predictions, labels)+ mm

    def wasserstein_term(self, predictions, labels):
        condition_t = torch.where(labels == self.succ_val, -1, 1)
        probs = nn.functional.softmax(predictions, dim=1)[:, self.succ_val]
        score1 = torch.dot(probs, condition_t.type(torch.float32))
        score1= score1*self.rep_batch
        score1 =0.5 * (1 + torch.tanh(self.mu * score1 + self.wbeq))
        return score1


if __name__ =='__main__':
    import numpy as np
    l0 =loss_manager(torch.tensor(8,1,2))
    x1v = np.asarray([[3.2, -1.3, 4.3, 2.1], [2.1, -0.8, 2.5, 1.1], [-0.6, 1.9, -3.2, 2.3], [-0.4, -2.1, 0.9, 2.23],
                      [-3.2, -1.3, 4.2, 2.1], [-1.1, -0.9, 2.5, 1.1], [-0.6, -1.9, 2.3, 1.1], [-0.4, -2.1, 3.1, 2.23]])
    print(x1v.shape)
    x2 = torch.from_numpy(x1v).type(torch.float32)
    y = torch.tensor([2, 2, 2, 2, 1, 2, 2, 0])
    print ( x2)
    print (y)
    l0.wasserstein_term(x2,y)
    xx= l0.regulation_term(x2,y)
    print ("sss=",l0.crit(x2,y))
    print(xx)
    print ("ok")