from os.path import join
import json
from pathlib import Path
import os

BNG_USER = f"{str(Path.home())}/Documents/BeamNG.research"
BNG_HOME = f"{str(Path.home())}/Desktop/beamng/trunk" #os.environ['BNG_HOME']
GEN_RANDOM = 'GEN_RANDOM'
GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'
GEN_DIVERSITY = 'GEN_DIVERSITY'

SEG_LENGTH = 25
NUM_SPLINE_NODES =10
INITIAL_NODE = (0.0, 0.0, -28.0, 8.0)
ROAD_BBOX_SIZE = (-1000, 0, 1000, 1500)
EXECTIME = 0
INVALID = 0



TARGET_SIZE     = 6

# K-nearest
K = 1

GEN              = 1000
POPSIZE          = 10


FEATURES         = ["Curvature", "SegmentCount"] # Curvature, SDSteeringAngle, SegmentCount, MeanLateralPosition
NUM_CELLS        = 25


GOAL             = (22, 6)

# goal cell for white area 
# goal cell for gray area (20, 4)
# goal cell for dark area Curv-Turn (22, 6)

DIVERSITY_METRIC = "INPUT" 

META_FILE       = "../experiments/data/bng/DeepHyperion/meta.json"



RESEEDUPPERBOUND = 2
MAX_BUCKET_SIZE = 50
RUN_TIME = 36000
TARGET_THRESHOLD = 1

def to_json(folder):
    config = {
        'features': str(FEATURES),
        'pop size': str(POPSIZE),
        'diversity': str(DIVERSITY_METRIC),
        'archive size': str(TARGET_SIZE), 
        'target cell': str(GOAL)
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
