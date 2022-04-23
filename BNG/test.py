import random
import self_driving.beamng_individual as BeamNGIndividual
from self_driving.beamng_member import BeamNGMember
import self_driving.beamng_problem as BeamNGProblem
import self_driving.beamng_config as cfg
from sample import Sample
from self_driving.road_bbox import RoadBoundingBox
import json
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

_config = cfg.BeamNGConfig()
problem = BeamNGProblem.BeamNGProblem(_config)

# samples = []
# POPSIZE = 80
# for i in range(0, POPSIZE):
#     max_angle = random.randint(10,100)
#     road = problem.generate_random_member(max_angle)
#     ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
#     ind.seed = i
#     sample = Sample(ind)
#     samples.append(sample) 

# distances = []

# for (s1, s2) in combinations(samples, 2):
#     distance = s1.ind.m.distance(s2.ind.m)
#     distances.append(distance)

# distances.sort(key = float)
# print(len(distances))
# print(np.mean(distances))
# print(np.max(distances))
# print(np.min(distances))

# percent_50 = int(len(distances) * 0.5)
# percent_30 = int(len(distances) * 0.3)
# percent_20 = int(len(distances) * 0.2)
# percent_10 = int(len(distances) * 0.1)
# percent_5 = int(len(distances) * 0.05)


# print("50%:", distances[percent_50])
# print("30%:", distances[percent_30])
# print("20%:", distances[percent_20])
# print("10%:", distances[percent_10])
# print("5%:", distances[percent_5])

# import os

# dataset = "D://tara//Results//TOSEM//BNG//DeepHyperion-CS//MeanLateralPosition_Curvature//1//outputs//1007//archive"

# samples = []
# i = 0
# for subdir, dirs, files in os.walk(dataset):
#     dirs.sort()
#     # Consider only the files that match the pattern
#     for sample_file in sorted([os.path.join(subdir, f) for f in files if
#         # TODO This is bit hacky to list all the json files that should not match...
#                         (
#                                 f.endswith("simulation.full.json")
#                         )]):
#             with open(sample_file, 'r') as input_file:
#                 simulation = json.load(input_file)
#                 nodes = simulation["road"]["nodes"] 
#                 road = BeamNGMember(nodes, nodes, 20, RoadBoundingBox((-250, 0, 250, 500)))
#                 ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
#                 ind.seed = i
#                 sample = Sample(ind)
#                 samples.append(sample) 
#                 i += 1


# distances = []

# for (s1, s2) in combinations(samples, 2):
#     distance = s1.ind.m.distance(s2.ind.m)
#     distances.append(distance)


# distances.sort(key = float)
# print(len(distances))
# print(np.mean(distances))
# print(np.max(distances))
# print(np.min(distances))

# percent_50 = int(len(distances) * 0.5)
# percent_30 = int(len(distances) * 0.3)
# percent_20 = int(len(distances) * 0.2)
# percent_10 = int(len(distances) * 0.1)
# percent_5 = int(len(distances) * 0.05)
# percent_1 = int(len(distances) * 0.01)


# print("50%:", distances[percent_50])
# print("30%:", distances[percent_30])
# print("20%:", distances[percent_20])
# print("10%:", distances[percent_10])
# print("5%:", distances[percent_5])
# print("1%:", distances[percent_1])

import re

log_file = "D://tara//dh-focused-test-generator//BNG/logs//1-nsga2_-features_MeanLateralPosition-SegmentCount-diversity_INPUT//logs.txt"
features = ["MeanLateralPosition", "SegmentCount"] #["SegmentCount", "MeanLateralPosition"] # , "Curvature", "SDSteeringAngle"

list_sample_ids = []
sample_track = []
coords = []


with open(log_file) as f:
    lines = f.readlines()

    for line in [l for l in lines if ("evaluated" in l)]:
        print(line)
        # extract feature and performance data from log line
        # ind 5 with seed 1 and (2, 13, 100, 170) and distance 12.571428571428573 evaluated
        pattern = re.compile("ind ([\d\.]+) with seed ([\d\.]+) and \(([\d\.]+), ([\d\.]+), ([\d\.]+),([\d\.]+)\)")

        pattern2 = re.compile("[-]?\d*\.?\d+")
        sample = pattern.findall(str(line))[0]

        if sample not in list_sample_ids:
            SegmentCount = int(sample[2])
            Curvature = int(sample[3])
            SDSteeringAngle = int(sample[4])
            MeanLateralPosition = int(sample[5])

            list_sample_ids.append(sample)

            if "MeanLateralPosition" in features and "SegmentCount" in features:
                sample_track.append(sample)
                coords.append((MeanLateralPosition, SegmentCount))
            elif "MeanLateralPosition" in features and "SDSteeringAngle" in features:
                sample_track.append(sample)
                coords.append((MeanLateralPosition, SDSteeringAngle))
            elif "Curvature" in features and "SDSteeringAngle" in features:
                sample_track.append(sample)
                coords.append((Curvature, SDSteeringAngle))
            elif "MeanLateralPosition" in features and "Curvature" in features:
                sample_track.append(sample)
                coords.append((MeanLateralPosition, Curvature))



x_val = [x[0] for x in coords]
y_val = [x[1] for x in coords]




plt.xlabel(features[0])
plt.ylabel(features[1])

plt.scatter(x_val, y_val)
plt.show()           


timestamp = []
for i in range(0,len(x_val)):
    timestamp.append(i)

plt.cla()

fig = plt.figure() #adding figure
ax = plt.axes(projection="3d")
ax.plot(x_val, y_val, timestamp)
plt.show()
