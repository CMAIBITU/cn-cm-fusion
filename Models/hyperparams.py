


from .util import Option


def getHyper_atlas(atlas=None):
    hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,
            # "lr" : 1e-4,
            # "minLr" : 1e-5,
            # "maxLr" : 2e-4,

            # FOR BOLT
            "nOfLayers" : 4,
            'dim1': 116,
            'dim2': 400,
            "dim" : 400,
            'atlas_mode':3, # 1 只用atlas1，2 只用atlas2, 3 都用 
            'norm_each_layer':True,
            
            'state_count':8,
            # 'state_dim': 1024,
            

            "numHeads" : 36,
            "headDim" : 20,

            "windowSize" : 8, # stride = windowSize * shiftCoeff
            "shiftCoeff" : 4.0/8.0,            
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
            "focalRule" : "fixed", #expand
            # "focalRule" : "expand", #expand
            
            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.5,
            "attnDrop" : 0.5,
            "lambdaCons" :0.5,
            
            # extra for ablation study
            "pooling" : "cls", # ["cls", "gmp"]         
                
            #kl add
            'cls_win_proj1':{
                'count_of_layers': 2,
                'input_dim': 116,
                'mid_dim': 116,
                'output_dim': 200
           },
            
            'cls_win_proj2':{
                'count_of_layers': 2,
                'input_dim': 400,
                'mid_dim': 400,
                'output_dim': 200
           },
            # 'cs': False, 
            'cs_loss_weight': 0.2, 
            'cls_loss_weight': 1, 
            # 'use_right_mask': True,
            # 'cs_space_dim':200,
            # 'n_splits':10,      
            # 'n_splits2':10,
            # 'use_float16': False,
            # 'rep_aug': False,
            # 'rep_fc':{
            #     'count_of_layers': 1,
            #     'input_dim': 128,
            #     'mid_dim': 128, # 不能改
            #     'output_dim': 200
            # }, 
        }
    return Option(hyperDict)
