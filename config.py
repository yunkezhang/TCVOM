from yacs.config import CfgNode as CN

_C = CN()
_C.MODEL = 'vmn50'
_C.AGG_WINDOW = 9
_C.SYSTEM = CN()
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
# Specific random seed, -1 for random.
_C.SYSTEM.RANDOM_SEED = -1
_C.SYSTEM.OUTDIR = 'train_log'
_C.SYSTEM.EXP_SUFFIX = ''
_C.SYSTEM.CUDNN_BENCHMARK = True
_C.SYSTEM.CUDNN_DETERMINISTIC = False
_C.SYSTEM.CUDNN_ENABLED = True

_C.DATASET = CN()
# dataset path
_C.DATASET.PATH = ''
_C.DATASET.SUBSET = False

_C.TRAIN = CN()
# checkpoint to load
_C.TRAIN.LOAD_CKPT = ''
_C.TRAIN.LOAD_OPT = ''
_C.TRAIN.FREEZE_BACKBONE = False
# Batch size per GPU
_C.TRAIN.BATCH_SIZE_PER_GPU = 1
_C.TRAIN.VAL_BATCH_SIZE_PER_GPU =1
# Learning rate
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.LR_STRATEGY = 'const' # 'poly' or 'const'
_C.TRAIN.WEIGHT_DECAY = 1e-4
# Network input size
_C.TRAIN.TRAIN_INPUT_SIZE = (512, 512)
_C.TRAIN.VAL_INPUT_SIZE = (512, 512)
_C.TRAIN.MIN_EDGE_LENGTH = 1088

# optimizer type
_C.TRAIN.OPTIMIZER = 'adam'
# total optimization step
_C.TRAIN.TOTAL_STEPS = 50
_C.TRAIN.PRINT_FREQ = 10
_C.TRAIN.IMAGE_FREQ = 500

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`