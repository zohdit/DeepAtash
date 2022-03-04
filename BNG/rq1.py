
from ast import Lambda
import os
import io
import json
import csv
from re import sub
import tensorflow as tf
from evaluator import Evaluator
from config import BITMAP_THRESHOLD, EXPECTED_LABEL, MODEL
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import utils as us
from scipy import stats
from matplotlib.pyplot import boxplot
import statsmodels.stats.power as pw
from sample import Sample
import random

model = tf.keras.models.load_model(MODEL)
encoder = tf.keras.models.load_model("models/vae_encoder_test", compile=False)
evaluator = Evaluator()

move_range = 1
bitmaps_range = 4
orientation_range = 11

def eff_size_label(eff_size):
    if np.abs(eff_size) < 0.2:
        return 'negligible'
    if np.abs(eff_size) < 0.5:
        return 'small'
    if np.abs(eff_size) < 0.8:
        return 'medium'
    return 'large'


def calculate_effect_size(array1, array2):
    # boxplot(array1, array2, labels=["approach1", "approach2"])

    (t, p) = stats.wilcoxon(array1, array2)
    eff_size = (np.mean(array1) - np.mean(array2)) / np.sqrt((np.std(array1) ** 2 + np.std(array2) ** 2) / 2.0)                   
    # powe = pw.FTestAnovaPower().solve_power(effect_size=eff_size, nobs=len(array1) + len(array2), alpha=0.05)
    # nruns = pw.FTestAnovaPower().solve_power(effect_size=eff_size, power=0.8, alpha=0.05)

    return eff_size, p


def load_data_all(dst, features):
    inputs = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        for svg_path in [os.path.join(subdir, f) for f in files if f.endswith(".svg")]:
            if features in svg_path:  
                print(".", end='', flush=True)  

                if "ga_" in svg_path:
                    y2 = "GA"
                elif "nsga2" in svg_path:
                    y2 = "NSGA2"

                if "LATENT" in svg_path:
                    y1 = "LATENT"
                elif "INPUT" in svg_path:
                    y1 = "INPUT"
                elif "HEATMAP" in svg_path:
                    y1 = "HEATMAP"

                with open(svg_path, 'r') as input_file:
                    xml_desc = input_file.read()       
                            
                json_path = svg_path.replace(".svg", ".json")            
                with open(json_path) as jf:
                    json_data = json.load(jf)

                npy_path = svg_path.replace(".svg", ".npy") 
                image = np.load(npy_path)

                png_path = svg_path.replace(".svg", ".png")

                inputs.append([image, f"{y1}-{y2}", json_data['predicted_label']])

    return inputs


def load_data(dst, i, approach, div):
    inputs = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        if approach in subdir and div in subdir:
            for svg_path in [os.path.join(subdir, f) for f in files if f.endswith(".svg")]:
                if i in svg_path:  
                    print(".", end='', flush=True)  

                    with open(svg_path, 'r') as input_file:
                        xml_desc = input_file.read()       
                                
                    json_path = svg_path.replace(".svg", ".json")            
                    with open(json_path) as jf:
                        json_data = json.load(jf)

                    npy_path = svg_path.replace(".svg", ".npy") 
                    image = np.load(npy_path)

                    inputs.append([image, f"{div}-{approach}", json_data['predicted_label']])

    return inputs


def getImage(path):
    return OffsetImage(plt.imread(path))


def plot_tSNE(inputs, _folder, features, div):
    """
    This function computes diversity using t-sne
    """
    X, y, imgs = [], [], []
    for i in inputs:
        X.append(i[0].flatten())
        y.append(i[1])
        imgs.append(i[2])
    
    X = np.array(X)

    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    tsne = TSNE(n_components=2, verbose=1, perplexity=0.1, n_iter=3000)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
   
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        # palette=sns.color_palette("hls", n_colors=10),
        data=df,
        legend="full",
        alpha=0.3
    )
    rand = random.randint(0,1000)
    fig.savefig(f"{_folder}/tsne-diag-{features}-{div}-{rand}-0.1.pdf", format='pdf')

    return df


def compute_distances_tSNE_all(df):
    input_ga = []
    input_nsga2 = []

    latent_ga = []
    latent_nsga2 = []

    heatmap_ga = []
    heatmap_nsga2 = []

    
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        if row['y'] == "INPUT-GA":
            input_ga.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "INPUT-NSGA2":
            input_nsga2.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "LATENT-GA":
            latent_ga.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "LATENT-NSGA2":
            latent_nsga2.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "HEATMAP-GA":
            heatmap_ga.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "HEATMAP-NSGA2":
            heatmap_nsga2.append((row['tsne-2d-one'], row['tsne-2d-two']))


    distance_input_ga = compute_average_distance(input_ga)
    distance_input_nsga2 = compute_average_distance(input_nsga2)
    distance_latent_ga = compute_average_distance(latent_ga)
    distance_latent_nsga2 = compute_average_distance(latent_nsga2)
    distance_heatmap_ga = compute_average_distance(heatmap_ga)
    distance_heatmap_nsga2 = compute_average_distance(heatmap_nsga2)

    return distance_input_ga, len(input_ga), distance_input_nsga2, len(input_nsga2)\
        , distance_latent_ga, len(latent_ga), distance_latent_nsga2, len(latent_nsga2) \
        ,    distance_heatmap_ga, len(heatmap_ga),  distance_heatmap_nsga2, len(heatmap_nsga2)


def compute_distances_tSNE_vs_dh_and_dlf(df, div, approach):
    approach_data = []
    dlf_data = []
    deephyperion_data = []
    
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        if row['y'] == f"{div}-{approach}":
            approach_data.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "DLFuzz":
            dlf_data.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "DeepHyperion":
            deephyperion_data.append((row['tsne-2d-one'], row['tsne-2d-two']))


    distance_approach = compute_average_distance(approach_data)
    distance_deephyperion = compute_average_distance(deephyperion_data)
    distance_dlf = compute_average_distance(dlf_data)

    return  distance_approach, len(approach_data)\
        , distance_deephyperion, len(deephyperion_data) ,\
            distance_dlf, len(dlf_data)

   
def compute_distances_tSNE(df):
    inputs = []

    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        inputs.append((row['tsne-2d-one'], row['tsne-2d-two']))

    distance_inputs = compute_average_distance(inputs)

    if len(inputs) > 1: 
        return np.mean(distance_inputs), len(inputs)
    else:
        return 0, len(inputs)


def compute_average_distance(coords):
    distances = []
    for coord1 in coords:
        for coord2 in coords:
            distances.append(us.manhattan(coord1, coord2))
    
    return np.mean(distances)


def compute_targets_for_dh(dst, goal, features, threshold, metric):
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

            svg_path = json_path.replace(".json", ".svg")
            with open(svg_path, 'r') as input_file:
                xml_desc = input_file.read()
        

            sample = Sample(xml_desc, EXPECTED_LABEL, int(ind["seed"]))

            sample.predicted_label = ind["predicted_label"]

            if features == "Moves-Bitmaps":
                cell = (int(ind["features"]["moves"]/move_range), int(ind["features"]["bitmaps"]/bitmaps_range))

            if features == "Orientation-Bitmaps":
                cell = (int(ind["features"]["orientation"]/orientation_range), int(ind["features"]["bitmaps"]/bitmaps_range))

            if features == "Orientation-Moves":
                cell = (int(ind["features"]["moves"]/move_range), int(ind["features"]["orientation"]/orientation_range))
            
            sample.distance_to_target = us.manhattan(cell, goal)
            if sample.distance_to_target <= 1 and sample.is_misbehavior() == True:
                sample.compute_explanation()
                sample.compute_latent_vector(encoder)
                print(".", end='', flush=True)
                samples.append(sample)
                count += 1

    samples = sorted(samples, key=lambda x: x.distance_to_target)
    archive = []
    for sample in samples:
        if len(archive) == 0:
            archive.append(sample)
        elif all(us.get_distance_by_metric(a, sample, metric)> threshold for a in archive):
            archive.append(sample)

    target_samples = []
    for sample in archive:
        target_samples.append([sample.purified, f"DeepHyperion", sample.predicted_label])

    print("DeepHyperion", features,len(target_samples))
    return target_samples


def compute_targets_for_dlf(dst, goal, features, threshold, metric):


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

            xml_desc = (ind["xml_desc"][2:len(ind["xml_desc"])-1]).replace("<?xml version=\\'1.0\\' encoding=\\'utf8\\'?>\\n", "")
            
            
            sample = Sample(xml_desc, EXPECTED_LABEL, int(ind["seed"]))
            sample.purified = img
            evaluator.evaluate(sample, model)


            bitmaps = us.bitmap_count(sample, BITMAP_THRESHOLD)
            moves = us.move_distance(sample)
            orientation = us.orientation_calc(sample, 0)

            if features == "Moves-Bitmaps":
                cell = (int(moves/move_range), int(bitmaps/bitmaps_range))

            if features == "Orientation-Bitmaps":
                cell = (int(orientation/orientation_range), int(bitmaps/bitmaps_range))

            if features == "Orientation-Moves":
                cell = (int(moves/move_range), int(orientation/orientation_range))
            
            sample.distance_to_target = us.manhattan(cell, goal)
            if sample.distance_to_target <= 1 and sample.is_misbehavior() == True:
                sample.compute_explanation()
                sample.compute_latent_vector(encoder)
                print(".", end='', flush=True)
                samples.append((ind, img))
                count += 1

    samples = sorted(samples, key=lambda x: x.distance_to_target)
    archive = []
    for sample in samples:
        if len(archive) == 0:
            archive.append(sample)
        elif all(us.get_distance_by_metric(a, sample, metric)> threshold for a in archive):
            archive.append(sample)

    target_samples = []
    for sample in archive:
        target_samples.append([sample.purified, f"DLFuzz", sample.predicted_label])

    print("DLFuzz", features, len(target_samples))
    return target_samples

       
def find_best_div_approach():

    evaluation_area = ["target_cell_in_dark", "target_cell_in_gray"]

    for evaluate in evaluation_area:
        dst = f"Data/new-FTG-percentile-0.1/{evaluate}"
        meanfile = f"Results/report_mean_{evaluate}.csv"
        fw = open(meanfile, 'w')
        cf = csv.writer(fw, lineterminator='\n')

        # write the header
        # approaches = ["ga_", "nsga2_"]
        # diversities = ["INPUT", "LATENT", "HEATMAP"]

        feature_combinations = ["Moves-Bitmap", "Orientation-Bitmap", "Orientation-Moves"]
        cf.writerow(["Features", "Approach", "Mean distance", "Num of inputs"])


        for features in feature_combinations:
            list_distance_input_ga = []
            list_num_input_ga = []
            list_distance_input_nsga2 = []
            list_num_input_nsga2 = []
            list_distance_latent_ga = []
            list_num_latent_ga = []
            list_distance_latent_nsga2 = []
            list_num_latent_nsga2 = []
            list_distance_heatmap_ga = []
            list_num_heatmap_ga = []
            list_distance_heatmap_nsga2 = []
            list_num_heatmap_nsga2 = []

            for i in range(1, 11):
                inputs = []
                for subdir, dirs, files in os.walk(dst, followlinks=False):
                    if features in subdir and str(i)+"-" in subdir:
                        data_folder = subdir
                        inputs = inputs + load_data_all(data_folder, features)
                df = plot_tSNE(inputs, dst, features, "")
                distance_input_ga, num_input_ga, distance_input_nsga2, num_input_nsga2 \
                , distance_latent_ga, num_latent_ga, distance_latent_nsga2, num_latent_nsga2\
                , distance_heatmap_ga, num_heatmap_ga, distance_heatmap_nsga2, num_heatmap_nsga2 = compute_distances_tSNE_all(df)
                
                list_distance_input_ga.append(distance_input_ga)
                list_num_input_ga.append(num_input_ga)
                list_distance_input_nsga2.append(distance_input_nsga2)
                list_num_input_nsga2.append(num_input_nsga2)
                list_distance_latent_ga.append(distance_latent_ga)
                list_num_latent_ga.append(num_latent_ga)
                list_distance_latent_nsga2.append(distance_latent_nsga2)
                list_num_latent_nsga2.append(num_latent_nsga2)
                list_distance_heatmap_ga.append(distance_heatmap_ga)
                list_num_heatmap_ga.append(num_heatmap_ga)
                list_distance_heatmap_nsga2.append(distance_heatmap_nsga2)
                list_num_heatmap_nsga2.append(num_heatmap_nsga2)

            cf.writerow([features, "GA-INPUT", np.nanmean(list_distance_input_ga), np.nanmean(list_num_input_ga)])
            cf.writerow([features, "NSGA2-INPUT", np.nanmean(list_distance_input_nsga2), np.nanmean(list_num_input_nsga2)])
            cf.writerow([features, "GA-LATENT", np.nanmean(list_distance_latent_ga), np.nanmean(list_num_latent_ga)])
            cf.writerow([features, "NSGA2-LATENT", np.nanmean(list_distance_latent_nsga2), np.nanmean(list_num_latent_nsga2)])
            cf.writerow([features, "GA-HEATMAP", np.nanmean(list_distance_heatmap_ga), np.nanmean(list_num_heatmap_ga)])
            cf.writerow([features, "NSGA2-HEATMAP", np.nanmean(list_distance_heatmap_nsga2), np.nanmean(list_num_heatmap_nsga2)])


def compare_with_dh_dlf(approach, div, features, target_area):

    if div == "HEATMAP":
        threshold = 0.09
    elif div == "INPUT":
        threshold = 4.8
    elif div == "LATENT":
        threshold = 0.01

    meanfile = f"Results/report_{approach}-{div}-{target_area}_vs_dh_dlf.csv"
    fw = open(meanfile, 'w')
    cf = csv.writer(fw, lineterminator='\n')

    cf.writerow(["Features", "Approach", "Mean distance", "Num of inputs"])

    result_folder = "Results"


    with open(f"Results/stats_{approach}-{div}-{target_area}_vs_dh_dlf.txt", "w") as text_file1:
        for feature in features:
            list_distance_deephyperion = []
            list_num_deephyperion = []
            list_distance_dlf = []
            list_num_dlf = []
            list_distance_approach = []
            list_num_approach = []
            dst = f"Data/new-FTG-percentile-0.1/{target_area}/{feature[0]}"
            dst_dh = f"Data/DeepHyperion/{feature[0]}"
            dst_dlf = f"Data/DLFuzz"
            for i in range(1, 11):
                for subdir, dirs, files in os.walk(dst_dh, followlinks=False):
                    if str(i)+"-" in subdir and "all" in subdir:
                        print(subdir)
                        inputs_dh = compute_targets_for_dh(subdir, feature[1], feature[0], threshold, div)
                
                inputs_focused = load_data(dst, str(i)+"-", approach, div)
                
                for subdir, dirs, files in os.walk(dst_dlf, followlinks=False):
                    if str(i)+"-" in subdir:
                        print(subdir)
                        inputs_dlf = compute_targets_for_dlf(dst_dlf, feature[1], feature[0], threshold, div)
                        
                if len(inputs_dh+inputs_dlf+inputs_focused) > 1:
                    df = plot_tSNE(inputs_dh+inputs_dlf+inputs_focused, result_folder, feature[0], div)
                    distance_approach, num_approach, distance_deephyperion, num_deephyperion, distance_dlf, num_dlf = compute_distances_tSNE_vs_dh_and_dlf(df, div, approach)
                else:
                    num_approach = len(inputs_focused)
                    num_deephyperion = len(inputs_dh)
                    num_dlf = len(inputs_dlf)

                if num_approach <= 1:
                    distance_approach = 0
                if num_deephyperion <= 1:
                    distance_deephyperion = 0
                if num_dlf <= 1:
                    distance_dlf = 0

                list_distance_dlf.append(distance_dlf)
                list_num_dlf.append(num_dlf)
                list_distance_approach.append(distance_approach)
                list_num_approach.append(num_approach)
                list_distance_deephyperion.append(distance_deephyperion)
                list_num_deephyperion.append(num_deephyperion)
            

            cf.writerow([feature[0], "DeepHyperion", np.mean(list_distance_deephyperion), np.mean(list_num_deephyperion)])
            cf.writerow([feature[0], "DLFuzz", np.mean(list_distance_dlf), np.mean(list_num_dlf)])
            cf.writerow([feature[0], f"{approach}-{div}", np.mean(list_distance_approach), np.mean(list_num_approach)])
            

            eff_size_distance, p_value_distance = calculate_effect_size(list_distance_approach, list_distance_deephyperion)
            eff_size_num, p_value_num = calculate_effect_size(list_num_approach, list_num_deephyperion)

            text_file1.write(f"{approach}-{div} vs DeepHyperion")
            text_file1.write(f"{feature[0]}\n")
            text_file1.write(f"Mean Distance: Cohen effect size = {eff_size_distance} ({eff_size_label(eff_size_distance)}); Wilcoxon p-value =  {p_value_distance}\n")
            text_file1.write(f"Num of inputs: Cohen effect size = {eff_size_num} ({eff_size_label(eff_size_num)}); Wilcoxon p-value =  {p_value_num}\n")

            eff_size_distance, p_value_distance = calculate_effect_size(list_distance_approach, list_distance_dlf)
            eff_size_num, p_value_num = calculate_effect_size(list_num_approach, list_num_dlf)

            text_file1.write(f"{approach}-{div} vs DLFuzz")
            text_file1.write(f"{feature[0]}\n")
            text_file1.write(f"Mean Distance: Cohen effect size = {eff_size_distance} ({eff_size_label(eff_size_distance)}); Wilcoxon p-value =  {p_value_distance}\n")
            text_file1.write(f"Num of inputs: Cohen effect size = {eff_size_num} ({eff_size_label(eff_size_num)}); Wilcoxon p-value =  {p_value_num}\n")



if __name__ == "__main__": 

    # dark cells
    # mov-lum (8,100) or-lum (160, 60), move-or (0, 160)
    find_best_div_approach()

    # feature and target
    # features = [("Moves-Bitmaps", (10/move_range, 140/bitmaps_range)), ("Orientation-Bitmaps",(160/orientation_range, 60/bitmaps_range)), ("Orientation-Moves",(0/move_range, 160/orientation_range))]
    
  
    # compare_with_dh_dlf("nsga2", "INPUT", features, "target_cell_in_dark")
    # compare_with_dh_dlf("nsga2", "HEATMAP", features, "target_cell_in_dark")
    # compare_with_dh_dlf("nsga2", "LATENT", features, "target_cell_in_dark")


    features = [("Moves-Bitmaps", (10/move_range, 124/bitmaps_range)), ("Orientation-Bitmaps",(-80/orientation_range, 40/bitmaps_range)), ("Orientation-Moves",(12/move_range, 90/orientation_range))]
   
    compare_with_dh_dlf("nsga2", "INPUT", features, "target_cell_in_gray")
    compare_with_dh_dlf("nsga2", "HEATMAP", features, "target_cell_in_gray")
    compare_with_dh_dlf("nsga2", "LATENT", features, "target_cell_in_gray")