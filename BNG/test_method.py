from datetime import datetime
import json
import sys
import random
from deap import base, creator, tools
import logging as log
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras


# local
from core.config import Config
import self_driving.beamng_config as cfg
import self_driving.beamng_individual as BeamNGIndividual
from archive import Archive
from config import POPSIZE, to_json
import self_driving.beamng_problem as BeamNGProblem
from config import GOAL, FEATURES, POPSIZE, RESEEDUPPERBOUND,\
     RUN_TIME, DIVERSITY_METRIC, META_FILE, TARGET_SIZE, INITIAL_POP, INITIAL_SEED, RUN_ID, APPROACH
import utils as us
from sample import Sample
from evaluator import Evaluator
from feature import Feature

evaluator = Evaluator()



lr_model_MLP = LinearRegression()
lr_model_StdSA = LinearRegression()
lr_model_behaviour = LinearRegression()

model_MLP=keras.models.Sequential([        
    keras.layers.Dense(24, input_shape=(24,), activation='relu'),       
    keras.layers.Dense(units=5,activation='relu'),
    keras.layers.Dense(units=1, activation="linear"),
])

model_StdSA=keras.models.Sequential([        
    keras.layers.Dense(24, input_shape=(24,), activation='relu'),       
    keras.layers.Dense(units=5,activation='relu'),
    keras.layers.Dense(units=1, activation="linear"),
])

model_behaviour=keras.models.Sequential([        
    keras.layers.Dense(24, input_shape=(24,), activation='relu'),       
    keras.layers.Dense(units=5,activation='relu'),
    keras.layers.Dense(units=1, activation="linear"),
])

def train_lrs(X, Y1, Y2, Y3):
    lr_model_MLP.fit(X, Y1)
    lr_model_StdSA.fit(X, Y2)
    lr_model_behaviour.fit(X, Y3)

def train_dnns(X_train, Y1_train, Y2_train, Y3_train):
    tf.keras.backend.clear_session()

    model_MLP.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model_MLP.fit(X_train, Y1_train, batch_size=32)

    model_StdSA.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model_StdSA.fit(X_train, Y2_train, batch_size=32)

    model_behaviour.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model_behaviour.fit(X_train, Y3_train, batch_size=32)

def generate_initial_pop(problem, features, goal):
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    samples = []
    for id in range(1, 81):
        max_angle = random.randint(10,100)
        road = problem.generate_random_member(max_angle)
        ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
        ind.seed = id
        sample = Sample(ind)
        samples.append(sample)    

        evaluator.evaluate(sample.ind)

        sample.features = {
                    "segment_count": us.segment_count(sample),
                    "sd_steering_angle": us.sd_steering(sample),
                    "mean_lateral_position": us.mean_lateral_position(sample),
                    "curvature": us.curvature(sample) 
        }

        b = tuple()
    
        for ft in features:
            i = ft.get_coordinate_for(sample)
            if i != None:
                b = b + (i,)
            else:
                # this individual is out of range and must be discarded
                sample.distance_to_target = np.inf
        
        sample.coordinate = b
        sample.distance_to_target = us.manhattan(b, goal)


        # training a regressor
        nodes = [[item[0], item[1]] for item in sample.ind.m.control_nodes]
        X.append(np.array(nodes).flatten())
        Y1.append(sample.features["mean_lateral_position"])
        Y2.append(sample.features["sd_steering_angle"])
        Y3.append(sample.ind.m.oob_ff)


    train_dnns(np.array(X), np.array(Y1), np.array(Y2), np.array(Y3))

    X1 = []
    y1_test = []
    y2_test = []
    y3_test = []

    for id in range(0, 20):
        max_angle = random.randint(10,100)
        road = problem.generate_random_member(max_angle)
        ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
        ind.seed = id
        sample = Sample(ind)
        samples.append(sample)    

        evaluator.evaluate(sample.ind)

        sample.features = {
                    "segment_count": us.segment_count(sample),
                    "sd_steering_angle": us.sd_steering(sample),
                    "mean_lateral_position": us.mean_lateral_position(sample),
                    "curvature": us.curvature(sample) 
        }

        nodes = [[item[0], item[1]] for item in sample.ind.m.control_nodes]
        X1.append(np.array(nodes).flatten())
        # X1.append(np.array([sample.features["segment_count"], sample.features["curvature"]]))
        y1_test.append(int(sample.features["mean_lateral_position"]))
        y2_test.append(int(sample.features["sd_steering_angle"]))
        y3_test.append(sample.ind.m.oob_ff)
        log.info(f"ind {sample.id} with seed {sample.seed} and ({sample.features['segment_count']}, {sample.features['curvature']}, {sample.features['sd_steering_angle']}, {sample.features['mean_lateral_position']}), performance {sample.ind.m.oob_ff} and distance {sample.distance_to_target} evaluated")

    y1_prediction = model_MLP.predict(np.array(X1))
    y2_prediction = model_StdSA.predict(np.array(X1))
    y3_prediction = model_behaviour.predict(np.array(X1))

    print(y1_prediction)
    print(y2_prediction)
    print(y3_prediction)

    # from sklearn.metrics import r2_score
    # print(f"score lr MLP: {r2_score(np.array(y1_test),y1_prediction)}")
    # print(f"score lr StdSA: {r2_score(np.array(y2_test),y2_prediction)}")
    # print(f"score lr behaviour: {r2_score(np.array(y3_test),y3_prediction)}")

    score = model_MLP.evaluate(np.array(X1), np.array(y1_test))

    print("Test loss MLP:", score[0])
    print("Test accuracy:", score[1])

    score = model_StdSA.evaluate(np.array(X1), np.array(y2_test))

    print("Test loss StdSA:", score[0])
    print("Test accuracy:", score[1])

    score = model_behaviour.evaluate(np.array(X1), np.array(y3_test))

    print("Test loss beahviour:", score[0])
    print("Test accuracy:", score[1])




def generate_features(meta_file):

    features = []
    with open(meta_file, 'r') as f:
        meta = json.load(f)["features"]
        

    if "Curvature" in FEATURES:
        f1 = Feature("curvature",meta["curvature"]["min"], meta["curvature"]["max"], "curvature", 25)
        features.append(f1)
    if "SegmentCount" in FEATURES:
        f2 = Feature("segment_count",meta["segment_count"]["min"], meta["segment_count"]["max"], "segment_count", meta["segment_count"]["max"]+1)
        features.append(f2)
    if "MeanLateralPosition" in FEATURES:
        f3 = Feature("mean_lateral_position", meta["mean_lateral_position"]["min"], meta["mean_lateral_position"]["max"], "mean_lateral_position", 25)
        features.append(f3)
    if "SDSteeringAngle" in FEATURES:
        f4 = Feature("sd_steering_angle",meta["sd_steering_angle"]["min"], meta["sd_steering_angle"]["max"], "sd_steering", 25)
        features.append(f4)
    return features




if __name__ == "__main__": 

    start_time = datetime.now()
    # run = sys.argv[1]
    name = f"logs/{RUN_ID}-{APPROACH}_-features_{FEATURES[0]}-{FEATURES[1]}-diversity_{DIVERSITY_METRIC}"
    
    Path(name).mkdir(parents=True, exist_ok=True)

    to_json(name)
    log_to = f"{name}/logs.txt"

    # Setup logging
    us.setup_logging(log_to)
    print("Logging results to " + log_to)

    features = generate_features(META_FILE)
    _config = cfg.BeamNGConfig()
    _config.name = name
    problem = BeamNGProblem.BeamNGProblem(_config)      

    generate_initial_pop(problem, features, GOAL)

    

    
