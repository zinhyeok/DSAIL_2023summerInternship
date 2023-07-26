import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


#MF base model r = q*p    
class MF(BaseModel):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        ##number of latent factors is hyperparameter, approximately 10~50 in this case
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users+1, num_factors)
        self.Q = nn.Embedding(num_items+1, num_factors)
        # self.user_bias = nn.Embedding(num_users, 1)
        # self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user, item):
        P_u = self.P(user)
        Q_i = self.Q(item)
        # b_u = self.user_bias(user)
        # b_i = self.item_bias(item)
        outputs = torch.sum(P_u * Q_i, dim=1)
        # outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
    

