
from email.mime import image
from itertools import combinations
import os
import tensorflow as tf
print(tf.config.experimental.list_physical_devices('GPU'))
from config import EXPECTED_LABEL, MODEL, BITMAP_THRESHOLD
import vectorization_tools
from sample import Sample
import json
from evaluator import Evaluator
import numpy as np
from utils import move_distance, bitmap_count, orientation_calc, euclidean, manhattan
import matplotlib.pyplot as plt
from explainer import explain_integrated_gradiant
import render
import vectorization_tools
from sample import Sample
import rasterization_tools
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from VAEH import sampling
from utils import heatmap_reshape

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
    (_, _), (x_test, y_test) = mnist.load_data()
    model = tf.keras.models.load_model(MODEL)
    encoder = tf.keras.models.load_model("models/vaeh_encoder", compile=False)
    evaluator = Evaluator()

    samples = []
    for seed in range(len(x_test)):
        if y_test[seed] == EXPECTED_LABEL:
            xml_desc = vectorization_tools.vectorize(x_test[seed])
            sample = Sample(xml_desc, EXPECTED_LABEL, seed)
            sample.compute_explanation()

            sample.compute_heatmap_latent_vector(encoder)
            
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
            d = euclidean(sample1.heatmap_latent_vector, sample2.heatmap_latent_vector)
            distances.append(d)
        
    distances.sort(key = float)

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

    print(np.mean(distances))
    print(max(distances))

# heatmap latent
# 10%: 0.3068403
# 5%: 0.20410825
# 1% 0.08747974
# 0.1% 0.026336411
# 0.01% 0.008508243
# 1.2128794
# 5.5969477

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
    # compute_threshold()

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (x_test, y_test) = mnist.load_data()
    model = tf.keras.models.load_model(MODEL)

    for seed in range(300,len(x_test)):
        if y_test[seed] == 5:
            xml_desc1 = vectorization_tools.vectorize(x_test[seed])
            x1 = rasterization_tools.rasterize_in_memory(xml_desc1)
            # R1 = explain_integrated_gradiant(x1)
            break;

    decoder = tf.keras.models.load_model("models/vae_decoder", compile=False)
    encoder = tf.keras.models.load_model("models/vae_encoder", compile=False)
    ae_type = "VAE"



    plt.imsave("mnist_test_original_2.png", np.reshape(x1, (28,28)))

    image_original = np.expand_dims(x1, -1).astype("float32") / 255
    image_original = np.where(image_original > 0.5, 1, image_original)
    
    # image_original = heatmap_reshape(x1)


    # z_mean, z_log_var, _ = encoder.predict(np.reshape(image_original, (-1, 28*28, )))
    z_mean, z_log_var, _ = encoder.predict(image_original)
    sampled_zs = sampling([z_mean, z_log_var])

    reconstructed = decoder.predict(sampled_zs)

    decoded_imgs_reshape = np.reshape(reconstructed, (28,28))
    plt.imsave("mnist_test_reconstructed_2.png", decoded_imgs_reshape)