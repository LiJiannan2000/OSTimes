from yacs.config import CfgNode as CN

config = CN()
config.NUM_WORKERS = 6
config.PRINT_FREQ = 5
config.VALIDATION_INTERVAL = 5
config.OUTPUT_DIR = 'experiments'
config.SEED = 12345

config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.DATASET = CN()
config.DATASET.ROOT = 'DATA/zhengdayi_all_large'
config.DATASET.DEF_ROOT = 'DATA/def_length/all'
config.DATASET.TEST_ROOT = 'DATA/BraTS20_all'
config.DATASET.TEST_DEF_ROOT = 'DATA/BraTS20_radiomics_deformation'
config.DATASET.input_channel = 4
config.DATASET.QUEUE_LENGTH = 300
config.DATASET.SAMPLES_PER_VOLUME = 10


config.MODEL = CN()
config.MODEL.NAME = 'OSPrediction'
config.MODEL.USE_PRETRAINED = False
config.MODEL.EXTRA = CN(new_allowed=True)
config.MODEL.INPUT_SIZE = [182, 218, 182]
config.MODEL.FEATURE_DEF_SIZE = 240
config.MODEL.NUM_HEADS = 4
config.MODEL.NUM_LAYERS = 2
config.MODEL.DROPOUT_RATE = 0.1
config.MODEL.ATTN_DROPOUT_RATE = 0.1

config.TRAIN = CN()
config.TRAIN.logdir = 'runs'
config.TRAIN.LR = 0.00015
config.TRAIN.WEIGHT_DECAY = 1e-4
config.TRAIN.BATCH_SIZE = 16
config.TRAIN.NUM_BATCHES = 250
config.TRAIN.EPOCH = 100
config.TRAIN.DEVICES = [0, 1, 2, 3]

config.INFERENCE = CN()
config.INFERENCE.BATCH_SIZE = 16

config.TFM = CN()
# The number of blocks in the model.
config.TFM.num_layers = 10
# The number of attention heads used in the attention layers of the model.
config.TFM.num_heads = 16
# The number of key-value heads for implementing attention.
config.TFM.num_kv_heads = 16
# The hidden size of the model.
config.TFM.hidden_size = 2016
# The dimension of the MLP representations.
config.TFM.intermediate_size = 2016
# The number of head dimensions.
config.TFM.head_dim = 126
# The epsilon used by the rms normalization layers.
config.TFM.rms_norm_eps = 1e-6

