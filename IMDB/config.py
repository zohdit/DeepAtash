# Make sure that any of this properties can be overridden using env.properties
from lib2to3.pgen2.token import NAME
from multiprocessing.pool import RUN
import os
from os.path import join
import json


# GA Setup
POPSIZE          = 100
NGEN             = 50000

RUN_TIME          = 3600

MODEL            = "models/imdb_michael_new.h5"

FEATURES        = ["PosCount", "VerbCount"] #PosCount NegCount WordCount


# target cell in dark poscount, negcount: (12, 56)
# target cell in dark negcount, verbcount: (67, 76)
# target cell in dark poscount, verbcount: (26, 228)

GOAL            = (26, 228)

XAI_METHOD      = "IG"

NUM_CELLS       = 25
RUN             = 1
NAME            = f"RUN_{RUN}_{POPSIZE}_{FEATURES[0]}-{FEATURES[1]}_{RUN_TIME}"

poscount_range  = 7
negcount_range   = 7
verbcount_range = 15

K               = 1
MAX_BUCKET_SIZE = 50
EXPECTED_LABEL  = 1 # 0 or 1
MUTLOWERBOUND    = 0.01
MUTUPPERBOUND    = 0.6

INITIAL_POP = 'seeded'

ORIGINAL_SEEDS = "starting_seeds_pos.txt"
DIVERSITY_METRIC = "INPUT" # "INPUT" "HEATMAP" "LATENT"
ARCHIVE_THRESHOLD = 217 # 217 for INPUT, 0.01 for LATENT, 0.09 IG HEATMAP
TARGET_THRESHOLD = 1


RESEEDUPPERBOUND = 10

VOCAB_SIZE = 2000

def to_json(folder):
    config = {
        'popsize': str(POPSIZE),
        'model': str(MODEL),
        'runtime': str(RUN_TIME),
        'features': str(FEATURES), 
        'expected label': str(EXPECTED_LABEL),
        'model': str(MODEL),
        'features': str(FEATURES),
        'diversity': str(DIVERSITY_METRIC),
        'archive threshold': str(ARCHIVE_THRESHOLD), 
        'target cell': str(GOAL), 
        'poscount_range': poscount_range,
        'negcount_range': negcount_range,
        'verb_range': verbcount_range     
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
