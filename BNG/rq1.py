
from ast import Lambda
import os
import io
import json
import csv

import matplotlib
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

from self_driving.road_bbox import RoadBoundingBox
from self_driving.beamng_member import BeamNGMember
import self_driving.beamng_problem as BeamNGProblem
import self_driving.beamng_individual as BeamNGIndividual
import self_driving.beamng_config as cfg


mlp_range = 2
curv_range = 1
sdstd_range = 7
turncnt_range = 1

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
        for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
            if features in json_path and "10h" in json_path:  
                print(".", end='', flush=True)  

                if "ga_" in json_path:
                    y2 = "GA"
                elif "nsga2" in json_path:
                    y2 = "NSGA2"


                y1 = "INPUT"
        
                with open(json_path) as jf:
                    json_data = json.load(jf)

                inputs.append([json_data["sample_nodes"], f"{y1}-{y2}", json_data['misbehaviour']])

    return inputs


def load_data(dst, i, approach, div):
    inputs = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        if approach in subdir and div in subdir:
            for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                if i in json_path  and "10h" in json_path:  
                    print(".", end='', flush=True)  

                    if "ga_" in json_path:
                        y2 = "GA"
                    elif "nsga2" in json_path:
                        y2 = "NSGA2"


                    y1 = "INPUT"
           
                    with open(json_path) as jf:
                        json_data = json.load(jf)


                    inputs.append([json_data["sample_nodes"], f"{y1}-{y2}", json_data['misbehaviour']])

    return inputs


def getImage(path):
    return OffsetImage(plt.imread(path))


def plot_tSNE(inputs, _folder, features, div):
    """
    This function computes diversity using t-sne
    """


    X, y = [], []
    for i in inputs:
        X.append(list(matplotlib.cbook.flatten(i[0])))
        # X.append(i[0])
        y.append(i[1])
    
    X = np.array(X)

    print("\n")
    print(X.shape)

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

    
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        if row['y'] == "INPUT-GA":
            input_ga.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "INPUT-NSGA2":
            input_nsga2.append((row['tsne-2d-one'], row['tsne-2d-two']))


    distance_input_ga = compute_average_distance(input_ga)
    distance_input_nsga2 = compute_average_distance(input_nsga2)


    return distance_input_ga, len(input_ga), distance_input_nsga2, len(input_nsga2)


def compute_distances_tSNE_vs_dh(df, div, approach):
    approach_data = []
    deephyperion_data = []
    
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        if row['y'] == f"{div}-{approach}":
            approach_data.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "DeepHyperion":
            deephyperion_data.append((row['tsne-2d-one'], row['tsne-2d-two']))


    distance_approach = compute_average_distance(approach_data)
    distance_deephyperion = compute_average_distance(deephyperion_data)


    return  distance_approach, distance_deephyperion, 

   
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
            distances.append(us.euclidean(coord1, coord2))
    
    return np.mean(distances)


def compute_targets_for_dh(dst, goal, features, threshold, metric):
    count = 0
    samples = []
    _config = cfg.BeamNGConfig()
    _config.name = ""
    problem = BeamNGProblem.BeamNGProblem(_config)
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        for json_path in [os.path.join(subdir, f) for f in files if
                        (
                            f.startswith("info_") and
                            f.endswith(".json")
                        )]:

            with open(json_path) as jf:
                ind_dict = json.load(jf)
            
            sim_path = subdir + "/simulation.full.json"

            with open(sim_path) as jf:
                sim = json.load(jf)

            nodes = sim["road"]["nodes"]
            bbox_size=(-250, 0, 250, 500)
            road = BeamNGMember(nodes, nodes, 20, RoadBoundingBox(bbox_size))
            ind = BeamNGIndividual.BeamNGIndividual(road, _config)

            misbehaviour = ind_dict["misbehaviour"]
            # ind.seed = ind_dict['seed']
            sample = Sample(ind)
            sample.misbehaviour = misbehaviour



            if features == "MeanLateralPosition_Curvature":
                cell = (int(ind_dict["features"]["mean_lateral_position"]/mlp_range), int(ind_dict["features"]["curvature"]/curv_range))

            if features == "MeanLateralPosition_SegmentCount":
                cell = (int(ind_dict["features"]["mean_lateral_position"]/mlp_range), int(ind_dict["features"]["segment_count"]/turncnt_range))

            if features == "Curvature_SDSteeringAngle":
                cell = (int(ind_dict["features"]["curvature"]/curv_range), int(ind_dict["features"]["sd_steering"]/sdstd_range))

            
            sample.distance_to_target = us.manhattan(cell, goal)

            if sample.distance_to_target <= 2 and misbehaviour == True:

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
        # print(sample.ind.m.control_nodes)
        target_samples.append([sample.ind.m.control_nodes, f"DeepHyperion", sample.misbehaviour])

    print("DeepHyperion", features, len(target_samples))
    return target_samples


       
def find_best_div_approach():

    evaluation_area = ["target_cell_in_dark"] #, "target_cell_in_gray"]

    for evaluate in evaluation_area:
        dst = f"logs/FTG"
        meanfile = f"Results/report_mean_{evaluate}.csv"
        fw = open(meanfile, 'w')
        cf = csv.writer(fw, lineterminator='\n')

        # write the header
        # approaches = ["ga_", "nsga2_"]
        # diversities = ["INPUT", "LATENT", "HEATMAP"]

        feature_combinations = ["MeanLateralPosition_Curvature", "MeanLateralPosition_Curvature", "Curvature_SDSteeringAngle"]
        cf.writerow(["Features", "Approach", "Mean distance", "Num of inputs"])


        for features in feature_combinations:
            list_distance_input_ga = []
            list_num_input_ga = []
            list_distance_input_nsga2 = []
            list_num_input_nsga2 = []


            for i in range(1, 6):
                inputs = []
                for subdir, dirs, files in os.walk(dst, followlinks=False):
                    if features in subdir and str(i)+"-" in subdir:
                        data_folder = subdir
                        inputs = inputs + load_data_all(data_folder, features)
                df = plot_tSNE(inputs, dst, features, "")
                distance_input_ga, num_input_ga, distance_input_nsga2, num_input_nsga2 = compute_distances_tSNE_all(df)
                
                list_distance_input_ga.append(distance_input_ga)
                list_num_input_ga.append(num_input_ga)
                list_distance_input_nsga2.append(distance_input_nsga2)
                list_num_input_nsga2.append(num_input_nsga2)


            cf.writerow([features, "GA-INPUT", np.nanmean(list_distance_input_ga), np.nanmean(list_num_input_ga)])
            cf.writerow([features, "NSGA2-INPUT", np.nanmean(list_distance_input_nsga2), np.nanmean(list_num_input_nsga2)])


def compare_with_dh(approach, div, features, target_area):

    if div == "INPUT":
        threshold = 10.0


    meanfile = f"Results/report_{approach}-{div}-{target_area}_vs_dh.csv"
    fw = open(meanfile, 'w')
    cf = csv.writer(fw, lineterminator='\n')

    cf.writerow(["Features", "Approach", "Mean distance", "Num of inputs"])

    result_folder = "Results"


    with open(f"Results/stats_{approach}-{div}-{target_area}_vs_dh.txt", "w") as text_file1:
        for feature in features:
            list_distance_deephyperion = []
            list_num_deephyperion = []

            list_distance_approach = []
            list_num_approach = []
            dst = f"logs/FTG/{feature[0]}"
            dst_dh = f"DeepHyperion/{feature[0]}"

            print(dst_dh)

            for i in range(1, 6):
                for subdir, dirs, files in os.walk(dst_dh, followlinks=False):
                    if str(i)+"-dh" in subdir and "beamng_nvidia_runner" in subdir:
                        inputs_dh = compute_targets_for_dh(subdir, feature[1], feature[0], threshold, div)
                        break
                
                inputs_focused = load_data(dst, str(i)+"-", approach, div)
                       
                if len(inputs_dh+inputs_focused) > 1:
                    df = plot_tSNE(inputs_dh+inputs_focused, result_folder, feature[0], div)
                    distance_approach, distance_deephyperion = compute_distances_tSNE_vs_dh(df, div, approach)

                    num_approach = len(inputs_focused)
                    num_deephyperion = len(inputs_dh)

                if num_approach <= 1:
                    distance_approach = 0
                if num_deephyperion <= 1:
                    distance_deephyperion = 0

                list_distance_approach.append(distance_approach)
                list_num_approach.append(num_approach)
                list_distance_deephyperion.append(distance_deephyperion)
                list_num_deephyperion.append(num_deephyperion)
            

            cf.writerow([feature[0], "DeepHyperion", np.mean(list_distance_deephyperion), np.mean(list_num_deephyperion)])
            cf.writerow([feature[0], f"{approach}-{div}", np.mean(list_distance_approach), np.mean(list_num_approach)])
            

            # eff_size_distance, p_value_distance = calculate_effect_size(list_distance_approach, list_distance_deephyperion)
            # eff_size_num, p_value_num = calculate_effect_size(list_num_approach, list_num_deephyperion)

            # text_file1.write(f"{approach}-{div} vs DeepHyperion")
            # text_file1.write(f"{feature[0]}\n")
            # text_file1.write(f"Mean Distance: Cohen effect size = {eff_size_distance} ({eff_size_label(eff_size_distance)}); Wilcoxon p-value =  {p_value_distance}\n")
            # text_file1.write(f"Num of inputs: Cohen effect size = {eff_size_num} ({eff_size_label(eff_size_num)}); Wilcoxon p-value =  {p_value_num}\n")



if __name__ == "__main__": 

    # dark cells
    # mov-lum (8,100) or-lum (160, 60), move-or (0, 160)
    # find_best_div_approach()

    # feature and target
    features = [("MeanLateralPosition_SegmentCount", (156/mlp_range, 4/turncnt_range)), ("MeanLateralPosition_Curvature",(167/mlp_range, 20/curv_range)), ("Curvature_SDSteeringAngle",(22/curv_range, 75/sdstd_range))]
    
  
    compare_with_dh("nsga2", "INPUT", features, "target_cell_in_dark")

