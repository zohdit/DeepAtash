from os.path import join
import json

# Dataset
EXPECTED_LABEL   = 5
# K-nearest
K = 1

GEN              = 1000
POPSIZE          = 100
IMG_SIZE         = 28
NUM_CLASSES      = 10
MODEL            = 'models/cnnClassifierTest.h5'
BITMAP_THRESHOLD = 0.5
FEATURES         = ["Orientation", "Moves"]#, "Orientation"]
NUM_CELLS        = 25

XAI_METHOD       = "IG" # "IG" or "CEM"
GOAL             = (24, 10) #(2, 140) 
# goal cell for white area (25, 35) (-30, 22) (3, -130) 
# goal cell for gray area (10, 124) (-80, 40) (12, 90) 
# goal cell with highest prob of failure mov-lum (10, 140) or-lum (160, 60), move-or (0, 160)
DIVERSITY_METRIC = "HEATMAP" # "INPUT" "HEATMAP" "LATENT"
ARCHIVE_THRESHOLD =  0.09 # 4.8 for INPUT, 0.01 for LATENT, 0.09 IG HEATMAP

RESEEDUPPERBOUND = 10
MAX_BUCKET_SIZE = 50
RUN_TIME = 3600

# mutation operator probability
MUTOPPROB        = 0.5
MUTOFPROB        = 0.5
MUTUPPERBOUND    = 0.6
MUTLOWERBOUND    = 0.01

def to_json(folder):
    config = {
        'label': str(EXPECTED_LABEL),
        'image size': IMG_SIZE,
        'num classes' : NUM_CLASSES,
        'model': str(MODEL),
        'features': str(FEATURES),
        'pop size': str(POPSIZE),
        'diversity': str(DIVERSITY_METRIC),
        'archive threshold': str(ARCHIVE_THRESHOLD), 
        'target cell': str(GOAL)
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
