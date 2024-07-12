
datasetDetailsDict = {

    "hcpRest" : {
        "datasetName" : "hcpRest",
        "targetTask" : "gender",
        "taskType" : "classification",
        "nOfClasses" : 2,        
        "dynamicLength" : 600,
        "foldCount" : 5,
        "atlas" : "schaefer7_400",
        "nOfEpochs" : 20,
        "batchSize" : 32       
    },

    "hcpTask" : {
        "datasetName" : "hcpTask",
        "targetTask" : "taskClassification",
        "nOfClasses" : 7,
        "dynamicLength" : 150,
        "foldCount" : 5,
        "atlas" : "schaefer7_400",
        "nOfEpochs" : 20,
        "batchSize" : 16
    },

    "abide1" : {
        # "datasetName" : "abide1_2atlas",
        "datasetName" : "abide1",
        "targetTask" : "disease",
        "nOfClasses" : 2,        
        # "dynamicLength" : 60,
        "dynamicLength" : 90,
        "foldCount" : 10,
        # "atlas" : "schaefer7_400",
        # "atlas": "aal-leida-tr2",
        # 'atlas': 'aal-tr2-0',
        # 'atlas':'cc200, sch400',
        'atlas': 'aal, sch400',
        # 'atlas': 'sch400',
        "nOfEpochs" : 20,
        "batchSize" : 32,
        'check': True,
        'no_0':False,
        
        'train_mul_factor': 1,
        'test_mul_factor':  1,
        
        'save': False,
        'ckpt_dir':'./ckpt/tmp/none/'
          
    },
    

 }



