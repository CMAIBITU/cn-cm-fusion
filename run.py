from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys
import wandb

from datetime import datetime

if(not "utils" in os.getcwd()):
    sys.path.append("../../../")


from Models.util import Option, calculateMetric
# from utils import Option, calculateMetric

from Models.model import Model
from Dataset.dataset import getDataset
import kl
from kl import torch_log


def train(model, dataset, fold, nOfEpochs, dataset_test=None, group=''):
    dataLoader = dataset.getFold(fold, train=True)
    
    # run = wandb.init(project='Bolt', reinit=True,
    #                      group=group, tags=[f"aal", 'dynamic_len=60'], name= group + '_' + str(fold),
    #                      notes='no data augment, kl RandomCrop, tr=2,  和数据标准化, epoch=20')
    logger = torch_log.TensorBoardLogger('fusion', group=group, project='atlas', name= str(fold), add_time=True)
    

    
    for epoch in range(nOfEpochs):

            preds = []
            probs = []
            groundTruths = []
            losses = []
            cs_losses = []
            dataset.cur_epoch = epoch
            for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
                
                xTrain = data["timeseries"] # (batchSize, N, dynamicLength)
                yTrain = data["label"] # (batchSize, )
                sids = data['subjId']
                sids = [str(s) for s in sids.tolist()]
                # sids = [ for sid in sids]
                # NOTE: xTrain and yTrain are still on "cpu" at this point

                train_cs_loss, train_loss, train_preds, train_probs, yTrain = model.step(xTrain, yTrain, sids=sids, folder_k=fold, train=True, epoch=epoch)
                # if need_cs_loss:
                    
                train_cs_loss = train_cs_loss if isinstance(train_cs_loss, (int, float)) else  train_cs_loss.numpy()
                train_loss = train_loss if isinstance(train_loss, (int, float)) else train_loss.numpy()
                train_preds = train_preds.numpy()
                train_probs = train_probs.numpy()
                yTrain = yTrain.numpy()
                
                torch.cuda.empty_cache()

                preds.append(train_preds)
                probs.append(train_probs)
                groundTruths.append(yTrain)
                losses.append(train_loss)
                cs_losses.append(train_cs_loss)

            # preds = torch.cat(preds, dim=0).numpy()
            # probs = torch.cat(probs, dim=0).numpy()
            # groundTruths = torch.cat(groundTruths, dim=0).numpy()
            # losses = torch.tensor(losses).numpy()
            # cs_losses = torch.tensor(cs_losses).numpy()
            preds = np.concatenate(preds, axis=0)
            probs = np.concatenate(probs, axis=0)
            groundTruths = np.concatenate(groundTruths, axis=0)
            losses = np.stack(losses)
            cs_losses = np.stack(cs_losses)

            metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
            print("Train metrics : {}".format(metrics), losses[0], cs_losses[0])
            train_metrics = {'train/' + key: metrics[key] for key in metrics}
            with torch.no_grad():
                _,_,_,_, test_metrics = test(model, dataset_test, fold)  
                test_metrics = { 'val/' + key: test_metrics[key] for key in test_metrics}
            
            # wandb.log({**train_metrics, **test_metrics})                
            logger.log_dict({**train_metrics, **test_metrics}, step=epoch)
            # logger.log_dict({**train_metrics}, step=epoch)
    # wandb.finish()
    # 都是numpy
    return preds, probs, groundTruths, losses



def test(model, dataset, fold):

    dataLoader = dataset.getFold(fold, train=False)

    preds = []
    probs = []
    groundTruths = []
    losses = []        

    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

        xTest = data["timeseries"]
        yTest = data["label"]

        # NOTE: xTrain and yTrain are still on "cpu" at this point

        _, test_loss, test_preds, test_probs, yTest = model.step(xTest, yTest, train=False)
        
        torch.cuda.empty_cache()

        preds.append(test_preds)
        probs.append(test_probs)
        groundTruths.append(yTest)
        losses.append(test_loss)

    preds = torch.cat(preds, dim=0).numpy()
    probs = torch.cat(probs, dim=0).numpy()
    groundTruths = torch.cat(groundTruths, dim=0).numpy()
    loss = torch.tensor(losses).numpy().mean()          

    metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
    print("\n \n Test metrics : {}".format(metrics))                
    
    return preds, probs, groundTruths, loss, metrics
    


def run_atlas(hyperParams, datasetDetails, device="cuda"):


    # extract datasetDetails

    foldCount = datasetDetails.foldCount
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs


    dataset = getDataset(datasetDetails)
    dataset_test = getDataset(datasetDetails)

    # print(dataset.get_nOfTrains_perFold())
    # exit()
    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "nOfClasses" : datasetDetails.nOfClasses,
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs,
        'atlas':datasetDetails.atlas,
        'check':datasetDetails.check,
    })


    results = []
    all_test_metrics = []
    timestamp = kl.stuff.get_readable_time()
    for fold in range(foldCount):

        model = Model(hyperParams, details)
        # summary(model, (32, 60, 116))
        # exit()


        train_preds, train_probs, train_groundTruths, train_loss = train(model, dataset, fold, nOfEpochs, dataset_test, group=timestamp)   
        # train_preds, train_probs, train_groundTruths, train_loss = train(model, dataset, fold, nOfEpochs, None, group=timestamp)   

        test_preds, test_probs, test_groundTruths, test_loss, test_metrics = test(model, dataset, fold)
        
        torch.cuda.empty_cache()
        
        all_test_metrics.append(test_metrics)
        result = {

            "train" : {
                "labels" : train_groundTruths,
                "predictions" : train_preds,
                "probs" : train_probs,
                "loss" : train_loss
            },

            "test" : {
                "labels" : test_groundTruths,
                "predictions" : test_preds,
                "probs" : test_probs,
                "loss" : test_loss
            }

        }

        results.append(result)

            
        model.free()
        del model
        torch.cuda.empty_cache()

    for i, m in enumerate(all_test_metrics):
        print(i, m)
    return results


if __name__ == "__main__":
    seeds=[0]
    resultss = []
    datasetName = 'abide1'
    from Dataset.datasetDetails import datasetDetailsDict
    from Models.hyperparams import getHyper_atlas
    from Models.util import calculateMetrics, metricSummer, dumpTestResults
    datasetDetails = datasetDetailsDict[datasetName]
    hyperParams = getHyper_atlas(datasetDetails['atlas'])
    for i, seed in enumerate(seeds):
    
        # for reproducability
        torch.manual_seed(seed)
        print("Running the model with seed : {}".format(seed))
        results = run_atlas(hyperParams, Option({**datasetDetails,"datasetSeed":seed}), device="cuda:0")
        resultss.append(results)
        


    metricss = calculateMetrics(resultss) 
    meanMetrics_seeds, stdMetrics_seeds, meanMetric_all, stdMetric_all = metricSummer(metricss, "test")


    # now dump metrics
    dumpTestResults('fusion', hyperParams, 'atlas', datasetName, metricss)

    print("\n \ n meanMetrics_all : {}".format(meanMetric_all))
    print("stdMetric_all : {}".format(stdMetric_all))


