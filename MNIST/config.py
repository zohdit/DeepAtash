from os.path import join
import json

# Dataset
EXPECTED_LABEL   = 5
# K-nearest
K = 1

GEN              = 1000000
POPSIZE          = 100
IMG_SIZE         = 28
NUM_CLASSES      = 10
MODEL            = 'models/cnnClassifierTest.h5'
BITMAP_THRESHOLD = 0.5
FEATURES         = ["Moves", "Bitmaps"]#, "Orientation", "Moves", "Bitmaps"

TARGET_SIZE     = 50


NUM_CELLS        = 25

XAI_METHOD       = "IG" # "IG" or "CEM"
GOAL             = (11, 158) 



# goal cell for white area mov-lum (13, 165) or-lum (160, 60) move-or (1, -175)  
# goal cell for gray area mov-lum (10, 124) or-lum (-54, 18) move-or (11, -40) 
# goal cell for dark area mov-lum (11, 158) or-lum (-30, 16) move-or (0, 160)

DIVERSITY_METRIC = "LATENT" # "INPUT" "HEATMAP" "LATENT" "HEATMAP_LATENT"
ARCHIVE_THRESHOLD = 0.01 # 4.8 for INPUT, 0.01 for LATENT, 0.09 IG HEATMAP, 0.02 HEATMAP_LATENT
TARGET_THRESHOLD = 1


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
        'target cell': str(GOAL), 
        'move_range': move_range,
        'bitmaps_range': bitmaps_range,
        'orientation_range': orientation_range        
    }

    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
