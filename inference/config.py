from easydict import EasyDict as edict


class Config:
    model_version = 'GammaLightGAIndoor10'
    # dataset
    DATASET = edict()
    
    
    DATASET.PATCH_WIDTH = 96
    DATASET.PATCH_HEIGHT = 96

    DATASET.AUG_MODE = True
    DATASET.VALUE_RANGE = 255.0

    DATASET.SEED = 0

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 4
    DATALOADER.NUM_WORKERS = 4

    # model
    MODEL = edict()

    MODEL.DEVICE = 'cuda'

    MODEL.USE_PCP_LOSS = True
    MODEL.USE_STYLE_LOSS = False
    MODEL.PCP_LOSS_WEIGHT = 1.0
    MODEL.STYLE_LOSS_WEIGHT = 0
    MODEL.PCP_LOSS_TYPE = 'l1'  # l1 | l2 | fro
    MODEL.VGG_TYPE = 'vgg19'
    MODEL.VGG_LAYER_WEIGHTS = dict(conv3_4=1/8, conv4_4=1/4, conv5_4=1/2)  # before relu
    MODEL.NORM_IMG = False
    MODEL.USE_INPUT_NORM = True


    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.BASE_LR = 1e-5
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.MAX_ITER = 400000
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.BIAS_WEIGHT = 0.0

    # initialization
    CONTINUE_ITER = None
    INIT_MODEL = None


    

    # log and save
    LOG_PERIOD = 10
    SAVE_PERIOD = 10000

    # validation
    VAL = edict()
    VAL.PERIOD = 1000
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.MAX_NUM = 100
    VAL.SAVE_IMG = True
    VAL.TO_Y = True


config = Config()



