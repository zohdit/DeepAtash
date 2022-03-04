
from itertools import combinations
import os
import tensorflow as tf
from config import EXPECTED_LABEL, MODEL, BITMAP_THRESHOLD
import vectorization_tools
from sample import Sample
import json
from evaluator import Evaluator
import numpy as np
from utils import move_distance, bitmap_count, orientation_calc, euclidean, manhattan
import matplotlib.pyplot as plt


def _basic_feature_stats():
    return {
        'min' : np.PINF,
        'max' : np.NINF,
        'missing' : 0
    }

def compute_threshold():
    data = {}
    # Overall Count
    data['total'] = 0
    feature = ["moves", "orientation", "bitmaps"]
    data['features'] = {f: _basic_feature_stats() for f in feature}
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (x_test, y_test) = mnist.load_data()
    model = tf.keras.models.load_model(MODEL)
    encoder = tf.keras.models.load_model("models/vae_encoder_test", compile=False)
    evaluator = Evaluator()

    samples = []
    for seed in range(len(x_test)):
        if y_test[seed] == EXPECTED_LABEL:
            xml_desc = vectorization_tools.vectorize(x_test[seed])
            sample = Sample(xml_desc, EXPECTED_LABEL, seed)
            sample.compute_latent_vector(encoder)
            sample.compute_explanation()
            performance = evaluator.evaluate(sample, model)
            # if performance > 0:
            #     misbehaviour = False
            # else:
            #     misbehaviour = True

            # sample_dict = {
            #     "expected_label": str(EXPECTED_LABEL),
            #     "features": {
            #         "moves":  move_distance(sample),
            #         "orientation": orientation_calc(sample,0),
            #         "bitmaps": bitmap_count(sample, BITMAP_THRESHOLD)
            #     },
            #     "id": sample.id,
            #     "seed": seed,
            #     "performance": str(performance),
            #     "misbehaviour": misbehaviour
            # }
            samples.append(sample)
            # for k, v in data['features'].items():
            #     # TODO There must be a pythonic way of doing it
            #     if k not in sample_dict["features"].keys():
            #         v['missing'] += 1

            #     v['min'] = min(v['min'], sample_dict["features"][k])
            #     v['max'] = max(v['max'], sample_dict["features"][k])

            # filedest = "logs/"+str(seed)+".json"
            # with open(filedest, 'w') as f:
            #     (json.dump(sample_dict, f, sort_keys=True, indent=4))

            # plt.imsave("logs/"+str(seed)+'exp.png', sample.explanation, cmap='gray', format='png')

    # for feature_name, feature_extrema in data['features'].items():
    #     parsable_string_tokens = ["=".join(["name",feature_name])]
    #     for extremum_name, extremum_value in feature_extrema.items():
    #         parsable_string_tokens.append("=".join([extremum_name, str(extremum_value)]))
    #     print(",".join(parsable_string_tokens))


    # # mean of max
    distances = []

    for (sample1, sample2) in combinations(samples, 2):
            d = euclidean(sample1.latent_vector, sample2.latent_vector)
            distances.append(d)
        
    distances.sort(key = float)
    print(len(distances))

    percent_10 = int(len(distances) * 0.1)
    percent_5 = int(len(distances) * 0.05)
    percent_1 = int(len(distances) * 0.01)
    percent_01 = int(len(distances) * 0.001)
    percent_001 = int(len(distances) * 0.0001)

    print("10%:", distances[percent_10])
    print("5%:", distances[percent_5])
    print("1%", distances[percent_1])
    print("0.1%", distances[percent_01])
    print("0.01%", distances[percent_001])

    # print(min(distances))
    # print(max(distances))


def compute_targets(dst, goal, features):
    count = 0
    samples = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
            for json_path in [os.path.join(subdir, f) for f in files if
                            (
                                f.startswith("mbr") and
                                f.endswith(".json")
                            )]:
                
                with open(json_path) as jf:
                    ind = json.load(jf)

                npy_path = json_path.replace(".json", ".npy")
                img = np.load(npy_path)

                if features in json_path:
                    if features == "Moves-Bitmaps":
                        cell = (int(ind["features"]["moves"]/5), int(ind["features"]["bitmaps"]/10))
                    distance = manhattan(cell, goal)
                    if distance <= 1 and ind["misbehaviour"] == True:
                        samples.append((ind, img))
                        count += 1

    archive = []
    for sample in samples:
        if len(archive) == 0:
            archive.append(sample)
        elif  all(euclidean(a[1], sample[1])> 3 for a in archive):
            print(sample[0]["id"])
            archive.append(sample)

    

    print("count", len(archive))




if __name__ == "__main__":
    # dst = "DeepHyperion/Moves-Bitmaps/"
    # features = "Moves-Bitmaps"
    # goal = (0, 10) # (8, 100)
    # compute_targets(dst, goal, features)

    compute_threshold()

    # mnist = tf.keras.datasets.mnist
    # (X_train, y_train), (x_test, y_test) = mnist.load_data()
    # model = tf.keras.models.load_model(MODEL)

    # fives = [184, 213, 345, 4230, 8010, 40329, 40063]
    # dst = "logs/test-with-all-initial-5-nsga2_-features_Moves-Bitmaps-diversity_INPUT"
    # for five in fives:
    #     for subdir, dirs, files in os.walk(dst, followlinks=False):
    #             for json_path in [os.path.join(subdir, f) for f in files if
    #                             (
    #                                 f.startswith(f"mbr{five}") and
    #                                 f.endswith(".json")

    #                             )]:
                 
    #                 with open(json_path) as jf:
    #                     ind = json.load(jf)

    #                 img = np.load(json_path.replace(".json", ".npy"))
    #                 xml_desc = vectorization_tools.vectorize(x_test[int(ind["seed"])])
    #                 sample = Sample(xml_desc, EXPECTED_LABEL, int(ind["seed"]))
    #                 original_img = sample.purified

    #                 distance = euclidean(img, original_img)
    #                 print(five, ": ", distance)
