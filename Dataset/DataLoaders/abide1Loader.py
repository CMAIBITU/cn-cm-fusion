import torch
import numpy as np
import os
import sys
from kl.data import check_signal_roi_health, save_pickle, load_pickle
from kl.torch_data.sur.utils import read_all_data
import glob



datadir = "/root/kl2/code/tmp/BolTkl/Dataset/Data"


def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (N, T)
    """


    # remove subjects with dead rois
    # return True
    
    if(np.sum(np.sum(np.abs(roiSignal), axis=1) == 0) > 0):
        return False

    return True    

def abide1Loader(atlas, targetTask, sort=True, no_0=True, check=False, resample=True):
    if len(atlas.split(',')) == 2:
        atlas1, atlas2 = atlas.split(',')
        atlas1 = atlas1.strip()
        atlas2 = atlas2.strip()
        return abide1Loader2Atlas(atlas1, atlas2, sort=sort, check=check, no_0=no_0, resample=resample)
    """
        x : (#subjects, N)
        /data3/surrogate/abide/checked/cc200/tall_tr2
    """
    folder_suff = '_no0' if no_0 else ''
    check_str = 'checked' if check else 'unchecked'
    data_root = f'/data3/surrogate/abide/{check_str}/{atlas}/tall{folder_suff}'
    if resample:
        data_root = f'/data3/surrogate/abide/{check_str}/{atlas}/tall_tr2{folder_suff}'
    subs = read_all_data(path=data_root)
    
    x=[]
    y=[]
    sids=[]
    for sub in subs:
        # if 'ts' in sub:
        #     x.append(sub['ts_stand'].T)
        # else:
        #     x.append(sub['ts_stand'].T)
        x.append(sub['time_series_stand_all'].T)
        y.append(int(sub['label'].item()))
        sids.append(int(sub['sid'].item()))
        
    return x, y, sids, [],[],[]
    
    dataset = torch.load(datadir + "/dataset_abide_{}.save".format(atlas))
    if sort:
        dataset = sorted(dataset, key=lambda x: x['pheno']['subjectId'])
    x = []
    y = []
    subjectIds = []

    x0 = []
    y0 = []
    subjectIds0 = []
    
    for data in dataset:
        
        if(targetTask == "disease"):
            label = int(data["pheno"]["disease"]) - 1 # 0 for autism 1 for control
        # data["roiTimeseries"].shape: (L, C)
        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):
        # if check_signal_roi_health(data["roiTimeseries"]):
            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            # print('jjjjjjjjj',data['pheno']['subjectId'])
            x0.append(data["roiTimeseries"].T)
            y0.append(label)
            subjectIds0.append(int(data["pheno"]["subjectId"]))
            # exit()
    print(len(x))
    return x, y, subjectIds, x0, y0, subjectIds0


def read_sids(data_root):
    subs = read_all_data(path=data_root)
    return set([int(sub['sid'].item()) for sub in subs])

def abide1Loader2Atlas(atlas1, atlas2, sort=True, check=False, no_0=True, resample=False):
    check_str = 'checked' if check else 'unchecked'
    folder_suff = '_no0' if no_0 else ''
    data_root1 = f'/data3/surrogate/abide/{check_str}/{atlas1}/tall{folder_suff}'
    data_root2 = f'/data3/surrogate/abide/{check_str}/{atlas2}/tall{folder_suff}'
    if resample:
        data_root1 = f'/data3/surrogate/abide/{check_str}/{atlas1}/tall_tr2{folder_suff}'
        data_root2 = f'/data3/surrogate/abide/{check_str}/{atlas2}/tall_tr2{folder_suff}'
    sids_path = f'/root/kl2/code/tmp/BolTkl/cache/{atlas1}_{atlas2}_no0_{no_0}_sids.pkl'
    if os.path.exists(sids_path):
        sids = load_pickle(sids_path)
    else:
        sids1 = read_sids(data_root1)
        sids2 = read_sids(data_root2)
        if no_0:
            sids = sids1.intersection(sids2)
        else:
            sids = sids1.union(sids2)
        save_pickle(sids, sids_path)
        
    sids = list(sids)
    if sort:
        sids.sort()
    
    def read_one_subs(data_root):
        data = np.load(data_root)
        return data['time_series_stand_tp'].T, int(data['label'].item())
        # return data['time_series_stand_all'].T, int(data['label'].item())
    x = []
    y = []
    for sid in sids:
        label = None
        data = []
        data_flag = [True, True] # 是否有data
        data_path1 = os.path.join(data_root1, f'{sid}.npz')
        data_path2 = os.path.join(data_root2, f'{sid}.npz')
        for i, data_path in enumerate([data_path1, data_path2]):
            if os.path.exists(data_path):
                ts, label = read_one_subs(data_path)
                data.append(ts)
            else:
                data_flag[i] = False
                data.append(None)
        x.append(data)
        y.append(label)    
    return x, y, sids, [],[],[]
    

if __name__ == '__main__':
    x, y, sids, x0, y0, sids0 = abide1Loader('aal', 'disease', sort=False)
    print(sids)
    # x, y, sids, x0, y0, sids0 = abide1Loader('aal-tr2', 'disease')
    # print('============================')
    # tx, ty, tsids, tx0, ty0, tsids0 = abide1Loader('aal-tr2-0', 'disease')
    # # target = '51578'
    # target = 51578
    # for ts0, sid0 in zip(tx0, tsids0):
    #     if sid0 == target:
    #         break
    # for ts, sid in zip(x, sids):
    #     if sid == target:
    #         break
    # ts = ts.T
    # ts0 = ts0.T
    # for c in range(116):
    #     if np.all(ts0[:, c] == 0):
    #         print(c) # 101, 106
    
    # print(ts[:, 101])
    
    # print('==============')
    # print(ts[:, 106])
    
    # ts[:, 101] = 0
    # ts[:, 106] = 0
    
    # print(np.all(ts == ts0))
    # print(ts[:, 0] - ts0[:, 0])