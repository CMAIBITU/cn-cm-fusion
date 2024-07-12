import torch
from Models.bolT import BolT
import numpy as np
from einops import rearrange
from torch.nn import functional as F
from torch import nn
import einops
import info_nce
import math

def cs_proj_factory(count_of_layers, input_dim, output_dim, mid_dim=None):
    if count_of_layers == 0:
        return nn.Identity()
    elif count_of_layers == 1:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
    elif count_of_layers == 2:
        if mid_dim is None:
            mid_dim = input_dim
        return nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, output_dim),
        )
    else:
        raise NotImplementedError()     

class BolTProxy(nn.Module):
    def __init__(self, hyperParams, details):
        super().__init__()
        # self.atlas1_embeding = nn.Linear(hyperParams.dim1, hyperParams.dim)
        # self.atlas2_embeding = nn.Linear(hyperParams.dim2, hyperParams.dim)
        
        self.atlas1_embeding = nn.Identity()
        self.atlas2_embeding = nn.Identity()
        
        # self.atlas1_embeding = nn.Sequential(
        #     nn.Linear(hyperParams.dim1, hyperParams.dim1),
        #     nn.ReLU(),
        #     nn.Dropout(.5),
        #     nn.Linear(hyperParams.dim1, hyperParams.dim)
        # )
        # self.atlas2_embeding = nn.Sequential(
        #     nn.Linear(hyperParams.dim2, hyperParams.dim2),
        #     nn.ReLU(),
        #     nn.Dropout(.5),
        #     nn.Linear(hyperParams.dim2, hyperParams.dim)
        # )
        self.atlas2_embeding = nn.Identity()
        
        self.state_feature = nn.Parameter(torch.randn(1, hyperParams.dim, hyperParams.state_count))# 需要 N, 8, state_dim 
        # self.reconstruct_rep = nn.Linear(hyperParams.state_count, 1)# 其实可以用PCA求解, 但这里先简单使用reconstruction的方式做
        # einops.repeat(self.cls_token, '() () c -> b l c', b=z.shape[0], l=z.shape[1])#
        self.state_count = hyperParams.state_count
        self.encoder1 = BolT(hyperParams, details, hyperParams.dim1)
        self.encoder2 = BolT(hyperParams, details, hyperParams.dim2)
        
        self.entropy_alpha =  nn.Parameter(torch.zeros(1))
        self.entropy_beta =  nn.Parameter(torch.zeros(1))
        
        self.w1 = nn.Parameter(torch.zeros(1) + 1)
        # self.w2 = nn.Parameter(torch.zeros(1) + 1)
        # self.w3 = nn.Parameter(torch.zeros(1) + 1)


        # self.encoder1 = BolT(hyperParams, details, dim=hyperParams.dim1)
        # self.encoder2 = BolT(hyperParams, details, dim=hyperParams.dim2)
        # self.cs_proj = cs_proj_factory(**hyperParams.cls_win_proj)
        self.cs_proj1 = cs_proj_factory(**hyperParams.cls_win_proj1)
        # self.cs_proj2 = self.cs_proj1
        self.cs_proj2 = cs_proj_factory(**hyperParams.cls_win_proj2)
        # self.fc = torch.nn.Linear(hyperParams.dim * 2, 2)
        self.fc = torch.nn.Linear(hyperParams.dim1 + hyperParams.dim2, 2)
        self.fc1 = torch.nn.Linear(hyperParams.dim1, 2)
        # self.fc2 = self.fc1
        self.fc2 = torch.nn.Linear(hyperParams.dim2, 2)
        

        
    
    def forward(self, ts, atlas=1):
        ts = einops.rearrange(ts, 'n c l->n l c')
        if atlas == 1:
            x = self.atlas1_embeding(ts)
            x = einops.rearrange(x, 'n l c -> n c l')
            out = self.encoder1(x)
        else:
            x = self.atlas2_embeding(ts)
            x = einops.rearrange(x, 'n l c -> n c l')
            out = self.encoder2(x)
        
        return out
    
    def cal_state_weight(self, cls_win_proj):
        # minimize(state_weight * states - feature), svd is enought
        U,S,V = torch.pca_lowrank(cls_win_proj, q=self.state_count, center=True, niter=2)
        state_weight = torch.bmm(cls_win_proj, V)
        return state_weight
        
    
    def predict4analyse(self, ts, analysis=False):
        # logits, cls, roiSignals_layers, cls_layers = self.forward(ts)
        _, cls, roiSignals_layers, cls_layers = self.model(ts, analysis=analysis)
        cluster_rep_proj = self.proj_cluster(cls)
        yHat = self.fc(torch.mean(cluster_rep_proj, dim=1))
        return yHat, cls
        

def get_ans(y, y_hat):
    with torch.no_grad():
        y = torch.argmax(y, dim=-1)
        y_hat = torch.argmax(y_hat, dim=-1)
        return y == y_hat         

class Model():

    def __init__(self, hyperParams, details):

        self.hyperParams = hyperParams
        self.details = details
        # print(self.hyperParams.dict) 
        # print(self.details.dict)
        '''
        {'weightDecay': 0, 'lr': 0.0002, 'minLr': 2e-05, 'maxLr': 0.0004, 'nOfLayers': 4, 'dim': 200, 'numHeads': 36, 'headDim': 20, 'windowSize': 14, 'shiftCoeff': 0.42857142857142855, 'fringeCoeff': 2, 'focalRule': 'expand', 'mlpRatio': 1.0, 'attentionBias': True, 'drop': 0.5, 'attnDrop': 0.5, 'lambdaCons': 0.5, 'pooling': 'cls', 'cs': True, 'cs_loss_weight': 0.2, 'use_right_mask': False, 'cs_space_dim': 200, 'n_splits': 10, 'n_splits2': 10}
        {'device': 'cuda:0', 'nOfTrains': 771, 'nOfClasses': 2, 'batchSize': 64, 'nOfEpochs': 20}
        '''
        self.model = BolTProxy(hyperParams, details)
        
        # load model into gpu     
        self.model = self.model.to(details.device)
        
        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)#, weight = classWeights)
        self.cs_loss_win = info_nce.InfoNCE(temperature=0.1) # 因为可能存在负样本中存在正样本，temperature适合设置大一点：https://zhuanlan.zhihu.com/p/506544456
        self.cs_loss_sub = info_nce.InfoNCE()
        self.cs_loss_tp = info_nce.InfoNCE()

        params = list(self.model.parameters())
        
        # set optimizer
        self.optimizer = torch.optim.Adam(params, lr = hyperParams.lr, weight_decay = hyperParams.weightDecay)

        # set scheduler
        self.scheduler = None
        steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))        
        divFactor = hyperParams.maxLr / hyperParams.lr
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr, details.nOfEpochs * (steps_per_epoch), div_factor=divFactor, final_div_factor=finalDivFactor, pct_start=0.3)

        self.cs_loss_weight = hyperParams.cs_loss_weight
        self.cls_loss_weight = hyperParams.cls_loss_weight
        self.lambdaCons = hyperParams.lambdaCons
        self.atlas_mode = hyperParams.atlas_mode
    
    def rep_fc_factory(self, count_of_layers, input_dim, output_dim, mid_dim=None):
        if count_of_layers == 0:
            return nn.Identity()
        elif count_of_layers == 1:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                # Norm()
            )
        elif count_of_layers == 2:
            if mid_dim is None:
                mid_dim = input_dim
            return nn.Sequential(
                nn.Linear(input_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, output_dim),
            )
        else:
            raise NotImplementedError()     
             
        
    def free(self):
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        torch.cuda.empty_cache()
        
        
    def sim(self,p1, p2):
            #p1, p2: (N, nW, C)
        dim = p1.shape[-1]
        p1 = p1.reshape(-1, dim)
        p2 = p2.reshape(-1, dim)
        r = p1@p2.T
        r = r.pow(2)
        return r.mean()
        # r = torch.sum(r) / r.shape[0]
        # return r
        
    def sim_self(self, p1, p2):
        ss = (p1 * p2).sum(-1)
        ss = ss.pow(2)
        return ss.mean()
        
        
    def step(self, x, y, sids=None, folder_k=0, train=True, epoch=None):
        # epoch_pretrain = 90
        # if epoch is not None:
        #     if epoch < epoch_pretrain:
        #         self.cls_loss_weight = 0
        #     else:
        #         self.cls_loss_weight = self.hyperParams.cls_loss_weight
        #         self.cs_loss_weight = 0
        #         self.lambdaCons = 0
                
        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)

        """
        B = y.shape[0]
        # PREPARE INPUTS
        
        inputs1, y = self.prepareInput(x[0], y)
        inputs2, y = self.prepareInput(x[1], y)

        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
        else:
            self.model.eval()
        

        # yHat, cls = self.model(*inputs)
        # cls_win:(batchSize, nW, featureDim)
        loss_win_cs_sim = 0
        loss_win_cs = 0
        loss_sub_cs = 0 
        loss_sub_cs_sim = 0
        loss_tp_cs = 0
        if self.atlas_mode == 3:
                    
            _, cls_win1, _, _ = self.model(*inputs1, atlas=1)
            _, cls_win2, _, _ = self.model(*inputs2, atlas=2)
            
            # if epoch is not None and epoch >= epoch_pretrain:
            #     cls_win1 = cls_win1.detach()
            #     cls_win2 = cls_win2.detach()
            # one = torch.eye(21).to(cls_win2.device)
            # ts_dp1 = torch.bmm(cls_win1[:, :21], cls_win1[:, :21].transpose(-1, 1)) - one # 128,21,21
            # ts_dp2 = torch.bmm(cls_win2[:, :21], cls_win2[:, :21].transpose(-1, 1)) - one # 128,21,21
            
            nw = cls_win1.shape[1]
            
            cls_win1_proj = self.model.cs_proj1(cls_win1)
            # cls_win2_proj = self.model.cs_proj2(cls_win2)
            cls_win2_proj = self.model.cs_proj2(cls_win2.detach())
            
            cls_win1_proj = F.normalize(cls_win1_proj, dim=-1)
            cls_win2_proj = F.normalize(cls_win2_proj, dim=-1)
        
            # cls_win_proj_cat = torch.cat((cls_win1_proj, cls_win2_proj), dim=0)
            # state_proj_cat = self.model.cal_state_weight(cls_win_proj_cat)
            # state1_proj = self.model.cal_state_weight(state1_proj) # (batchSize, nW, state_count)
            # state2_proj = self.model.cal_state_weight(state2_proj)
            # state1_proj = state_proj_cat[:B]
            # state2_proj = state_proj_cat[B:]
            
            state1_proj = cls_win1_proj
            state2_proj = cls_win2_proj 
            
            # if train:
            #     if epoch % 2 == 0:
            #         state1_proj = state1_proj.detach()
            #     else:
            #         state2_proj = state2_proj.detach()
            
            rep = torch.cat((cls_win1.mean(dim=1), cls_win2.mean(dim=1)), dim=-1)
            # rep = torch.cat((cls_win1.mean(dim=1), cls_win2.mean(dim=1), ts_dp1.mean(-1), ts_dp2.mean(-1)), dim=-1)
            yHat = self.model.fc(rep)
            loss_cls = self.getLoss(yHat, y, torch.cat((cls_win1, cls_win2), dim=-1))   
            
            cls_win1_ = cls_win1
            rep1 = cls_win1_.mean(dim=1)
            yHat1 = self.model.fc1(rep1)
            loss_cls1 = self.getLoss(yHat1, y, cls_win1_)

            cls_win2_ = cls_win2
            rep2 = cls_win2_.mean(dim=1)
            yHat2 = self.model.fc2(rep2)
            loss_cls2 = self.getLoss(yHat2, y, cls_win2_)
            
            # yHat = yHat1 + yHat2
            # loss_cls = loss_cls1 + loss_cls2
    
            loss_cls = loss_cls + loss_cls1 * 1 + loss_cls2 * 1
            
            # yHat = yHat2
            
        elif self.atlas_mode == 1:
            _, cls_win, _, _ = self.model(*inputs1, atlas=1)
            rep = cls_win.mean(dim=1)
            yHat = self.model.fc1(rep)
            loss_cls = self.getLoss(yHat, y)
        else:
            _, cls_win, _, _ = self.model(*inputs2, atlas=2)
            rep = cls_win.mean(dim=1)
            yHat = self.model.fc2(rep)
            loss_cls = self.getLoss(yHat, y)
            
        loss_state_cs = 0
        if train and self.atlas_mode == 3:
            # neighbor = 4
            # diag = torch.eye(B*nw).cuda()
            # mask = torch.ones_like(diag)
            # mask = torch.tril(mask, -(neighbor + 1)) + torch.triu(mask, neighbor + 1) + diag
            
            dim = state1_proj.shape[-1]
            # loss_win_cs = self.cs_loss(state1_proj.reshape(-1, dim), 
            #                             state2_proj.reshape(-1, dim), mask_KL=mask)
            
            # loss_sub_cs = self.cs_loss(state1_proj.mean(dim=1), state2_proj.mean(dim=1))
            # loss_state_cs = loss_state_cs + loss_sub_cs 
            loss_win_cs = self.cs_loss_win(state1_proj.reshape(-1, dim), 
                                        state2_proj.reshape(-1, dim), no_positive=True, self_ratio=self.model.entropy_alpha, ratio_expand=None)
            # print(self.model.entropy_alpha)
            
            loss_win_cs_sim = self.sim(state1_proj, state2_proj)
            # loss_sub_cs = self.cs_loss(state1_proj.mean(dim=1), state2_proj.mean(dim=1), no_positive=False)
            loss_sub_cs = self.cs_loss_sub(state1_proj.mean(dim=1), state2_proj.mean(dim=1), no_positive=True, self_ratio=self.model.entropy_beta)
            loss_sub_cs_sim = self.sim(state1_proj.mean(dim=1), state2_proj.mean(dim=1))
            
            # N, L, C
            ts_dp1 = torch.bmm(state1_proj, state1_proj.transpose(-1, 1))# 128,21,21
            indices = torch.triu_indices(row=ts_dp1.shape[1], col=ts_dp1.shape[2], offset=1)
            ts_dp1 = ts_dp1[:,indices[0], indices[1]]
            ts_dp2 = torch.bmm(state2_proj, state2_proj.transpose(-1, 1))
            ts_dp2 = ts_dp2[:,indices[0], indices[1]]
            loss_tp_cs = self.cs_loss_tp(ts_dp1, ts_dp2, no_positive=False)
            
            # ts_dp1 = torch.bmm(cls_win1, cls_win1.transpose(-1, 1))# 128,21,21
            # indices = torch.triu_indices(row=ts_dp1.shape[1], col=ts_dp1.shape[2], offset=1)
            # ts_dp1 = ts_dp1[:,indices[0], indices[1]]
            # ts_dp2 = torch.bmm(cls_win2, cls_win2.transpose(-1, 1))
            # ts_dp2 = ts_dp2[:,indices[0], indices[1]]
            # loss_tp_cs = self.cs_loss(ts_dp1, ts_dp2)
            
            # loss_state_cs = loss_win_cs*2 + loss_win_cs_sim*2 + loss_sub_cs * 2 + loss_sub_cs_sim * 2 + loss_tp_cs * 2
            
            # loss_state_cs = loss_win_cs + loss_win_cs_sim + loss_sub_cs * 0 + loss_win_cs_sim * 0 + loss_tp_cs * 0
            # loss_state_cs = loss_win_cs * self.model.w1 + loss_sub_cs * self.model.w1 + loss_tp_cs * (2 - self.model.w1)
            loss_state_cs = loss_win_cs + loss_sub_cs * 2 + loss_tp_cs * 2

        
        loss = loss_cls + loss_state_cs * self.cs_loss_weight
        preds = yHat.argmax(1)
        probs = yHat.softmax(1)

        if(train):
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if(not isinstance(self.scheduler, type(None))):
                self.scheduler.step()            

        loss = loss.detach().to("cpu")
        # pos_one = 0 if loss_state_cs == 0 else loss_state_cs.detach().to("cpu")
        
        # pos_one = self.model.entropy_alpha.detach().to("cpu")
        pos_one = 0 if loss_win_cs == 0 else loss_win_cs.detach().to("cpu")
        
        loss_cls = 0 if loss_cls == 0 else loss_cls.detach().to("cpu")
        preds = preds.detach().to("cpu")
        probs = probs.detach().to("cpu")

        y = y.to("cpu")
        # cls = None # 有释放，不多
        
        torch.cuda.empty_cache()

        return pos_one, loss_cls, preds, probs, y
        # return loss_state_cs, loss_cls, preds, probs, y, {'alpha:': self.model.entropy_alpha.detach(), 'beta': self.model.entropy_beta.detach()}
        


    # HELPER FUNCTIONS HERE

    def prepareInput(self, x, y):

        """
            x = (batchSize, N, T)
            y = (batchSize, )

        """
        # to gpu now

        x = x.to(self.details.device)
        y = y.to(self.details.device)


        return (x, ), y

    # def getLoss(self, yHat, y):
    #     cross_entropy_loss = self.criterion(yHat, y)
    #     return cross_entropy_loss * self.cls_loss_weight
    
    def getLoss(self, yHat, y, cls=None):
        
        # cls.shape = (batchSize, #windows, featureDim)
        if cls is None:
            clsLoss = 0
        else:
            clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))

        # clsLoss = 0
        cross_entropy_loss = self.criterion(yHat, y)

        return cross_entropy_loss * self.cls_loss_weight + clsLoss * self.lambdaCons

