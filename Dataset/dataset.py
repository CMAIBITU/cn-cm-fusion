
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from random import shuffle, randrange
import numpy as np
import random

#from .DataLoaders.hcpRestLoader import hcpRestLoader
#from .DataLoaders.hcpTaskLoader import hcpTaskLoader
from .DataLoaders.abide1Loader import abide1Loader
from kl.augment import *



loaderMapper = {
    #"hcpRest" : hcpRestLoader,
    #"hcpTask" : hcpTaskLoader,
    "abide1" : abide1Loader,
}


def getDataset(options, swap_train_test=False):
    return SupervisedDataset(options, swap_train_test=swap_train_test)

class SupervisedDataset(Dataset):
    
    def __init__(self, datasetDetails, swap_train_test=False):

        self.batchSize = datasetDetails.batchSize
        self.dynamicLength = datasetDetails.dynamicLength
        self.foldCount = datasetDetails.foldCount

        self.seed = datasetDetails.datasetSeed
        # print('seed', self.seed)
        # exit()

        loader = loaderMapper[datasetDetails.datasetName]
        self.data, self.labels, self.subjectIds, self.data0, self.labels0, self.subjectIds0 = loader(datasetDetails.atlas, datasetDetails.targetTask, no_0=datasetDetails.no_0, check=datasetDetails.check)
        self.multi_atlas = len(datasetDetails.atlas.split(',')) > 1
        print('data count:', len(self.data))
        self.k = None
        # self.kFold = StratifiedKFold(datasetDetails.foldCount, shuffle=True, random_state=self.seed) if datasetDetails.foldCount is not None else None
        self.kFold = StratifiedKFold(datasetDetails.foldCount, shuffle=False) if datasetDetails.foldCount is not None else None
        # 其实就是 ts, label, sid
        # print(self.seed)
        # exit()
        random.Random(self.seed).shuffle(self.data)
        random.Random(self.seed).shuffle(self.labels)
        random.Random(self.seed).shuffle(self.subjectIds)
        # print(self.subjectIds)
        # exit()
        
        self.data_full = self.data + self.data0
        self.labels_full = self.labels + self.labels0
        self.subjectIds_full = self.subjectIds + self.subjectIds0

        self.targetData = None
        self.targetLabel = None
        self.targetSubjIds = None

        self.randomRanges = None

        self.trainIdx = None
        self.testIdx = None
        # kl add
        self.atlas = datasetDetails.atlas
        self.cur_epoch = 0
        self.use_full_data = False
        self.swap_train_test = swap_train_test
        
        self.train_mul_factor = datasetDetails.train_mul_factor
        self.test_mul_factor = datasetDetails.test_mul_factor
        
        self.train_augment_60 = Compose([
            # RandomCropAndPadding(60),
            # InterpTR2(p=0.0,a=-1, b=1),
            # Jitter(0.03),
            # Scaling(sigma=0.23),
            # Standardization(dim=0)
        ])

        self.test_augment = Compose([
                # CropAndPadding(0,90)
                # RandomCropAndPadding(datasetDetails.dynamicLength),
                # RandomCrop(90),
                # Standardization(dim=0)
        ])


    def __len__(self):
        return len(self.data) if isinstance(self.targetData, type(None)) else len(self.targetData)
    

    def get_nOfTrains_perFold(self):
        if(self.foldCount != None):
            if self.use_full_data:
                # print('full')
                return int(np.ceil(len(self.data) * (self.foldCount - 1) / self.foldCount)) + len(self.data0)   
            else:
                return int(np.ceil(len(self.data) * (self.foldCount - 1) / self.foldCount))           
        elif self.use_full_data:
            return len(self.data_full)
        else:
            return len(self.data)         

    def setFold(self, fold, train=True):

        if self.use_full_data and train:
            # print('train include channel=0 data')
            self.k = fold
            self.train = train


            if(self.foldCount == None): # if this is the case, train must be True
                trainIdx = list(range(len(self.data)))
            else:
                trainIdx, testIdx = list(self.kFold.split(self.data, self.labels))[fold]      
            
            # np.concatenate
            trainIdx = trainIdx.tolist() + list(range(len(self.data), len(self.data_full)))
            # self.trainIdx = trainIdx
            # self.testIdx = testIdx
            
            # print(len(self.trainIdx))

            random.Random(self.seed).shuffle(trainIdx)

            self.targetData = [self.data_full[idx] for idx in trainIdx] 
            self.targetLabels = [self.labels_full[idx] for idx in trainIdx] 
            self.targetSubjIds = [self.subjectIds_full[idx] for idx in trainIdx]
            # if(train and not isinstance(self.dynamicLength, type(None))):
            #     np.random.seed(self.seed+1)
            #     self.randomRanges = [[np.random.randint(0, self.data_full[idx].shape[-1] - self.dynamicLength) for k in range(9999)] for idx in trainIdx]
        else:
            # print('train exclude channel=0 data')
            self.k = fold
            self.train = train


            if(self.foldCount == None): # if this is the case, train must be True
                trainIdx = list(range(len(self.data)))
            else:
                trainIdx, testIdx = list(self.kFold.split(self.data, self.labels))[fold]
                
            # print('bolt:', testIdx)   
            # exit()   

            
            # print(trainIdx.tolist())
            # exit()
            if self.swap_train_test:
                tmp = trainIdx
                trainIdx = testIdx
                testIdx = tmp
            
            random.Random(self.seed).shuffle(trainIdx)
            trainIdx = trainIdx.tolist() * self.train_mul_factor
            testIdx = testIdx.tolist() * self.test_mul_factor
            self.trainIdx = trainIdx
            self.testIdx = testIdx
            
            self.targetData = [self.data[idx] for idx in trainIdx] if train else [self.data[idx] for idx in testIdx]
            self.targetLabels = [self.labels[idx] for idx in trainIdx] if train else [self.labels[idx] for idx in testIdx]
            self.targetSubjIds = [self.subjectIds[idx] for idx in trainIdx] if train else [self.subjectIds[idx] for idx in testIdx]
            
            
            
            # self.targetData = self.targetData
            # print(self.targetSubjIds)
            # print(fold)
            
            
            if(train and not isinstance(self.dynamicLength, type(None))):
                np.random.seed(self.seed+1)
                if self.multi_atlas:
                    self.randomRanges = [[np.random.randint(0, self.data[idx][0].shape[-1] - self.dynamicLength) for k in range(9999)] for idx in trainIdx]
                    self.pp = [[self.data[idx][0].shape[-1] for k in range(9999)] for idx in trainIdx]
                    print('hh')
                else:
                    self.randomRanges = [[np.random.randint(0, self.data[idx].shape[-1] - self.dynamicLength) for k in range(9999)] for idx in trainIdx]
        
        # if train:
        #     print('train set len:', len(self.targetData))
        
    def getFold(self, fold, train=True):
        
        self.setFold(fold, train)

        if(train):
            return DataLoader(self, batch_size=self.batchSize, shuffle=False)
        else:
            # return DataLoader(self, batch_size=self.batchSize, shuffle=False) 
                   
            return DataLoader(self, batch_size=1, shuffle=False)            

    def get_item_2_atlas(self, idx):
        subject = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]

        # normalize timeseries
        timeseries = [subject[0], subject[1]] # (numberOfRois, time)

        for i, ts in enumerate(timeseries):  
            timeseries[i] = (ts - np.mean(ts, axis=1, keepdims=True)) / np.std(ts, axis=1, keepdims=True)
            timeseries[i] = np.nan_to_num(ts, 0)

        # dynamic sampling if train
        if(self.train and not isinstance(self.dynamicLength, type(None))):
            # if(timeseries.shape[1] < self.dynamicLength):
            #     print(timeseries.shape[1], self.dynamicLength)
            samplingInit = self.randomRanges[idx].pop()
            # print('pop')
            for i, ts in enumerate(timeseries):  
                ts = ts[:, samplingInit : samplingInit + self.dynamicLength]
                ts = ts.T
                ts = self.train_augment_60(ts)
                ts = ts.T
                if ts.shape[1] < self.dynamicLength:
                    print('wrong.......', i, ts.shape[1], 'sid', subjId) 
                timeseries[i] = ts
            
        elif self.train is False:
            for i, ts in enumerate(timeseries):  
                ts = ts.T
                ts = self.test_augment(ts)
                ts = ts.T
                timeseries[i] = ts

        for i, ts in enumerate(timeseries):  
            timeseries[i] = ts.astype(np.float32)
            
        return {"timeseries" : timeseries, "label" : label, "subjId" : subjId}
        
    
    def __getitem__(self, idx):
        if self.multi_atlas:
            return self.get_item_2_atlas(idx)
        
        subject = self.targetData[idx]
        label = self.targetLabels[idx]
        subjId = self.targetSubjIds[idx]


        # normalize timeseries
        timeseries = subject # (numberOfRois, time)
        # timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)) / np.std(timeseries, axis=1, keepdims=True)
        # timeseries = np.nan_to_num(timeseries, 0)

        # dynamic sampling if train
        if(self.train and not isinstance(self.dynamicLength, type(None))):
            # if(timeseries.shape[1] < self.dynamicLength):
            #     print(timeseries.shape[1], self.dynamicLength)

            samplingInit = self.randomRanges[idx].pop()
            # print('pop')
            timeseries = timeseries[:, samplingInit : samplingInit + self.dynamicLength]
            
            timeseries = timeseries.T
            timeseries = self.train_augment_60(timeseries)
            timeseries = timeseries.T
        elif self.train is False:
            timeseries = timeseries.T
            timeseries = self.test_augment(timeseries)
            timeseries = timeseries.T

        return {"timeseries" : timeseries.astype(np.float32), "label" : label, "subjId" : subjId}







