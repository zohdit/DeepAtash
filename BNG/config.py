from os.path import join
import json
from pathlib import Path
import os

BNG_USER = f"{str(Path.home())}/Documents/BeamNG.research"
BNG_HOME = os.environ['BNG_HOME']
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

# Dataset
EXPECTED_LABEL   = 5
# K-nearest
K = 1

GEN              = 1000
POPSIZE          = 48


FEATURES         = ["MeanLateralPosition", "SegmentCount"] # Curvature, SDSteeringAngle
NUM_CELLS        = 25

GOAL             = (156, 4)
# goal cell for white area 
# goal cell for gray area 
# goal cell for dark area  MLP-TurnCnt (156, 4), MLP-StdSA (158,97), Curv-StdSA (22, 75)
DIVERSITY_METRIC = "INPUT" 
ARCHIVE_THRESHOLD =  35.0 

RESEEDUPPERBOUND = int(POPSIZE * 0.1)
MAX_BUCKET_SIZE = 50
RUN_TIME = 36000


def to_json(folder):
    config = {
        'features': str(FEATURES),
        'pop size': str(POPSIZE),
        'diversity': str(DIVERSITY_METRIC),
        'archive threshold': str(ARCHIVE_THRESHOLD), 
        'target cell': str(GOAL)
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
