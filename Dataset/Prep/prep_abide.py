import os

import nilearn as nil
import nilearn.datasets
import nilearn.image
from glob import glob
import pandas
import torch
from tqdm import tqdm
import numpy as np

# from .prep_atlas import prep_atlas
# from nilearn.input_data import NiftiLabelsMasker


datadir = "./Dataset/Data"
atlas='sch400'


def prep_abide(atlas):

    bulkDataDir = "{}/Bulk/ABIDE".format(datadir)

    # atlasImage = prep_atlas(atlas)



    # if(not os.path.exists(bulkDataDir)):
    #     nil.datasets.fetch_abide_pcp(data_dir=bulkDataDir, pipeline="cpac", band_pass_filtering=False, global_signal_regression=False, derivatives="func_preproc", quality_checked=True)

    dataset = []


    # temp = pandas.read_csv(bulkDataDir + "/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv").to_numpy()
    temp = pandas.read_csv('/root/kl2/data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv').to_numpy()
    phenoInfos = {}
    # print(temp)
    for row in temp:
        # print(row[2])
        # print(row[5])
        phenoInfos[str(row[2])] = {"site": row[5], "age" : row[9], "disease" : row[7], "gender" : row[10]}

    print("\n\nExtracting ROIS...\n\n")

    # for scanImage_fileName in tqdm(glob(bulkDataDir+"/ABIDE_pcp/cpac/nofilt_noglobal/*"), ncols=60):
        
    #     if(".gz" in scanImage_fileName):

    #         scanImage = nil.image.load_img(scanImage_fileName)
    #         roiTimeseries =  NiftiLabelsMasker(atlasImage).fit_transform(scanImage)

    #         subjectId = scanImage_fileName.split("_")[-3][2:]
            
    #         dataset.append({
    #             "roiTimeseries": roiTimeseries,
    #             "pheno": {
    #                 "subjectId" : subjectId, **phenoInfos[subjectId]
    #             }
    #         })
    
    if atlas == 'cc200':
        tr = 0
        if tr == 2:
            for scanImage_fileName in tqdm(glob('/data3/surrogate/abide/checked/aal/tall_tr2/*.npz'), ncols=60):
                
                if(".npz" in scanImage_fileName):

                    # scanImage = nil.image.load_img(scanImage_fileName)
                    # roiTimeseries =  NiftiLabelsMasker(atlasImage).fit_transform(scanImage)
                    # roiTimeseries = np.load(scanImage_fileName)
                    # subjectId = scanImage_fileName.split("_")[-3][2:]
                    data = np.load(scanImage_fileName)
                    roiTimeseries = data['ts_stand']
                    subjectId = data['sid'].item()
                    
                    dataset.append({
                        "roiTimeseries": roiTimeseries,
                        "pheno": {
                            "subjectId" : subjectId, **phenoInfos[subjectId]
                        }
                    })
        else:
            for scanImage_fileName in tqdm(glob('/root/kl2/data/ABIDE_pcp/cpac/filt_global/*.1D'), ncols=60):
                    
                if("cc200" in scanImage_fileName):

                    # scanImage = nil.image.load_img(scanImage_fileName)
                    # roiTimeseries =  NiftiLabelsMasker(atlasImage).fit_transform(scanImage)
                    roiTimeseries = np.loadtxt(scanImage_fileName, skiprows=0)
                    subjectId = scanImage_fileName.split("_")[-3][2:]
                    
                    dataset.append({
                        "roiTimeseries": roiTimeseries,
                        "pheno": {
                            "subjectId" : subjectId, **phenoInfos[subjectId]
                        }
                    })
    elif atlas == 'sch400':
        path = '/data3/surrogate/abide/checked/sch400/tall_no0/*.npz'
        for file in tqdm(glob(path), ncols=60):
            data = np.load(file)
            sid = data['sid'].item()

            dataset.append({
                    "roiTimeseries": data['ts'],
                    "pheno": {
                        "subjectId" : sid, **phenoInfos[sid]
                    }
                })
            
            
            # if(".npy" in scanImage_fileName):

            #     # scanImage = nil.image.load_img(scanImage_fileName)
            #     # roiTimeseries =  NiftiLabelsMasker(atlasImage).fit_transform(scanImage)
            #     roiTimeseries = np.load(scanImage_fileName)
            #     subjectId = scanImage_fileName.split("_")[-3][2:]
                
            #     dataset.append({
            #         "roiTimeseries": roiTimeseries,
            #         "pheno": {
            #             "subjectId" : subjectId, **phenoInfos[subjectId]
            #         }
            #     })
        
    else:
        for scanImage_fileName in tqdm(glob('/root/kl2/data/ABIDE_pcp/cpac/filt_global/*.npy'), ncols=60):
            
            if(".npy" in scanImage_fileName):

                # scanImage = nil.image.load_img(scanImage_fileName)
                # roiTimeseries =  NiftiLabelsMasker(atlasImage).fit_transform(scanImage)
                roiTimeseries = np.load(scanImage_fileName)
                subjectId = scanImage_fileName.split("_")[-3][2:]
                
                dataset.append({
                    "roiTimeseries": roiTimeseries,
                    "pheno": {
                        "subjectId" : subjectId, **phenoInfos[subjectId]
                    }
                })

    torch.save(dataset, datadir + "/dataset_abide_{}.save".format(atlas) )


if __name__ == '__main__':
   prep_abide(atlas)
    
