
import os
import json
import time

import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import utils as us
from sample import Sample


from self_driving.simulation_data import SimulationParams, SimulationData, SimulationDataRecord, SimulationInfo
from self_driving.road_bbox import RoadBoundingBox
from self_driving.vehicle_state_reader import VehicleState
from self_driving.beamng_member import BeamNGMember
from self_driving.decal_road import DecalRoad
import self_driving.beamng_problem as BeamNGProblem
import self_driving.beamng_individual as BeamNGIndividual
import self_driving.beamng_config as cfg
from config import TARGET_THRESHOLD, mlp_range, sdstd_range, curv_range, turncnt_range

num_cells = 100

def load_data_all(dst, features):
    inputs = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
            if features in json_path:  
                print(".", end='', flush=True)  

                if "ga_" in json_path:
                    y2 = "GA"
                elif "nsga2" in json_path:
                    y2 = "NSGA2"

                y1 = "INPUT"
        
                with open(json_path) as jf:
                    json_data = json.load(jf)

                inputs.append([json_data["sample_nodes"], f"{y2}-{y1}", json_data['misbehaviour']])

    return inputs


def load_data(dst, i, approach, div):
    inputs = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        if i+approach in subdir and div in subdir and "10h" in subdir:
            for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                    print(".", end='', flush=True)  
           
                    with open(json_path) as jf:
                        json_data = json.load(jf)

                    inputs.append([json_data["sample_nodes"], f"{approach}-{div}", json_data['misbehaviour']])
    print(len(inputs))
    return inputs


def plot_tSNE(inputs, _folder, features, div, ii=0):
    """
    This function computes diversity using t-sne
    """
    X, y = [], []
    for i in inputs:
        X.append(list(matplotlib.cbook.flatten(i[0])))
        y.append(i[1])

    
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
    fig.savefig(f"{_folder}/tsne-diag-{features}-{div}-{ii}-0.1.pdf", format='pdf')

    return df


def compute_distances_tSNE_all(df):
    input_ga = []
    input_nsga2 = []

    
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        if row['y'] == "GA-INPUT":
            input_ga.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "NSGA2-INPUT":
            input_nsga2.append((row['tsne-2d-one'], row['tsne-2d-two']))


    distance_input_ga = compute_average_distance(input_ga)
    distance_input_nsga2 = compute_average_distance(input_nsga2)


    list_data = [("GA", "Input", distance_input_ga, len(input_ga)), ("NSGA2", "Input", distance_input_nsga2, len(input_nsga2))]
    
    return list_data


def compute_average_distance(coords):
    distances_euc = []
    distances_manh = []
    for coord1 in coords:
        for coord2 in coords:
            distances_euc.append(us.euclidean(np.array(coord1), np.array(coord2)))
            distances_manh.append(us.manhattan(coord1, coord2))
    
    return (np.nanmean(distances_euc), np.nanmean(distances_manh))



def compute_tSNE_and_coverage_all(inputs, _folder, features, div, ii=0, num=10):

    distance_input_ga = []
    distance_input_nsga2 = []
    distance_latent_ga = []
    distance_latent_nsga2 = []
    distance_heatmap_ga = []
    distance_heatmap_nsga2 = []

    for index in range(0, num):
        df = plot_tSNE(inputs, _folder, features, div, ii)

        input_ga = []
        input_nsga2 = []

        latent_ga = []
        latent_nsga2 = []

        heatmap_ga = []
        heatmap_nsga2 = []

        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0

        
        df = df.reset_index()  # make sure indexes pair with number of rows
        for index, row in df.iterrows():
            if row['tsne-2d-one'] < min_x:
                min_x = row['tsne-2d-one']
            if row['tsne-2d-one'] > max_x:
                max_x = row['tsne-2d-one']

            if row['tsne-2d-two'] < min_y:
                min_y = row['tsne-2d-two']
            if row['tsne-2d-two'] > max_y:
                max_y = row['tsne-2d-two']

            range_xy = [(min_x, max_x), (min_y, max_y)]

            if row['y'] == "GA-INPUT":
                input_ga.append((row['tsne-2d-one'], row['tsne-2d-two']))
            if row['y'] == "NSGA2-INPUT":
                input_nsga2.append((row['tsne-2d-one'], row['tsne-2d-two']))
            if row['y'] == "GA-LATENT":
                latent_ga.append((row['tsne-2d-one'], row['tsne-2d-two']))
            if row['y'] == "NSGA2-LATENT":
                latent_nsga2.append((row['tsne-2d-one'], row['tsne-2d-two']))
            if row['y'] == "GA-HEATMAP":
                heatmap_ga.append((row['tsne-2d-one'], row['tsne-2d-two']))
            if row['y'] == "NSGA2-HEATMAP":
                heatmap_nsga2.append((row['tsne-2d-one'], row['tsne-2d-two']))


        distance_input_ga.append(compute_coverage_tSNE(input_ga, range_xy))
        distance_input_nsga2.append(compute_coverage_tSNE(input_nsga2, range_xy))
        distance_latent_ga.append(compute_coverage_tSNE(latent_ga, range_xy))
        distance_latent_nsga2.append(compute_coverage_tSNE(latent_nsga2, range_xy))
        distance_heatmap_ga.append(compute_coverage_tSNE(heatmap_ga, range_xy))
        distance_heatmap_nsga2.append(compute_coverage_tSNE(heatmap_nsga2, range_xy))

    list_data = [("GA", "Input", np.nanmean(distance_input_ga), len(input_ga)), ("NSGA2", "Input", np.nanmean(distance_input_nsga2), len(input_nsga2))]
                    
    return list_data


def compute_coverage_tSNE(coords, range):
    x_range = np.linspace(range[0][0], range[0][1], num_cells)
    y_range = np.linspace(range[1][0], range[1][1], num_cells)

    coverage_data = np.zeros(shape=(num_cells, num_cells), dtype=int)
    for coord in coords:
        x = np.digitize(coord[0], x_range, right=False) - 1
        y = np.digitize(coord[1], y_range, right=False) - 1
        coverage_data[x, y] += 1
    
    coverage_count = np.count_nonzero(coverage_data > 0)

    return  coverage_count


def find_best_div_approach(dst, feature_combinations):

    evaluation_area = ["target_cell_in_dark"]
    print(dst)

    for evaluate in evaluation_area:
        
        for features in feature_combinations:
            for i in range(1, 11):
                inputs = []
                for subdir, dirs, files in os.walk(dst, followlinks=False):
                    if features in subdir and str(i)+"-" in subdir and evaluate in subdir:
                        data_folder = subdir
                        inputs = inputs + load_data_all(data_folder, features)

                if len(inputs) > 1:
                    list_data = compute_tSNE_and_coverage_all(inputs, f"{dst}/{evaluate}/{features}", features, i)
                else:
                    list_data = [("GA", "Input", np.nan, 0), ("NSGA2", "Input",  np.nan, 0)]

                
                for data in list_data:
                    dict_data = {
                        "approach": data[0],
                        "diversity": data[1],
                        "run": i,
                        "test input count": data[3],
                        "features": features,
                        "avg tsne coverage": str(data[2])
                        # "euclidean sparseness": str(data[2][0]),
                        # "manhattan sparseness": str(data[2][1])
                    }

                    filedest = f"{dst}/{evaluate}/{features}/report_{data[0]}-{data[1]}_{i}.json"
                    with open(filedest, 'w') as f:
                        (json.dump(dict_data, f, sort_keys=True, indent=4))

def compute_targets_for_dh(dst, goal, features, threshold, metric):
    count = 0
    samples = []
    _config = cfg.BeamNGConfig()
    _config.name = ""
    problem = BeamNGProblem.BeamNGProblem(_config)
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        for json_path in [os.path.join(subdir, f) for f in files if
                        (
                            f.startswith("simulation") and
                            f.endswith(".json")
                        )]:

            with open(json_path) as jf:
                data = json.load(jf)

            

            nodes = data["road"]["nodes"]
            bbox_size=(-250, 0, 250, 500)
            member = BeamNGMember(data["control_nodes"], nodes, 20, RoadBoundingBox(bbox_size))
            records = data["records"]
            simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
            sim_name = simulation_id # member.config.simulation_name.replace('$(id)', simulation_id)
            simulation_data = SimulationData(sim_name)
            states = []
            for record in records:
                state = VehicleState(timer=record["timer"]
                                    , damage=record["damage"]
                                    , pos=record["pos"]
                                    , dir=record["dir"]
                                    , vel=record["vel"]
                                    , gforces=record["gforces"]
                                    , gforces2=record["gforces2"]
                                    , steering=record["steering"]
                                    , steering_input=record["steering_input"]
                                    , brake=record["brake"]
                                    , brake_input=record["brake_input"]
                                    , throttle=record["throttle"]
                                    , throttle_input=record["throttle_input"]
                                    , throttleFactor=record["throttleFactor"]
                                    , engineThrottle=record["engineThrottle"]
                                    , wheelspeed=record["engineThrottle"]
                                    , vel_kmh=record["engineThrottle"])

                sim_data_record = SimulationDataRecord(**state._asdict(),
                                                    is_oob=record["is_oob"],
                                                    oob_counter=record["oob_counter"],
                                                    max_oob_percentage=record["max_oob_percentage"],
                                                    oob_distance=record["oob_distance"])
                states.append(sim_data_record)

            simulation_data.params = SimulationParams(beamng_steps=data["params"]["beamng_steps"], delay_msec=int(data["params"]["delay_msec"]))
            simulation_data.control_nodes = data["control_nodes"]
            simulation_data.road = DecalRoad.from_dict(data["road"])
            simulation_data.info = SimulationInfo()
            simulation_data.info.start_time = data["info"]["start_time"]
            simulation_data.info.end_time = data["info"]["end_time"]
            simulation_data.info.elapsed_time = data["info"]["elapsed_time"]
            simulation_data.info.success = data["info"]["success"]
            simulation_data.info.computer_name = data["info"]["computer_name"]
            simulation_data.info.id = data["info"]["id"]

            simulation_data.states = states

            if len(states) > 0:
                member.distance_to_boundary = simulation_data.min_oob_distance()
                member.simulation = simulation_data
                misbehaviour = states[-1].is_oob
            else:
                misbehaviour = False
                break
            
            ind = BeamNGIndividual.BeamNGIndividual(member, _config)
            sample = Sample(ind)
            #us.is_oob(sample.ind.sample_nodes, sample.ind.member.simulation.states)
            
            sample.misbehaviour = misbehaviour

            mlp = us.mean_lateral_position(sample)
            stdsa = us.sd_steering(sample)
            curv = us.curvature(sample)
            turncnt = us.segment_count(sample)

            if features == "MeanLateralPosition_Curvature":
                cell = (int(mlp/mlp_range), int(curv/curv_range))

            if features == "MeanLateralPosition_SegmentCount":
                cell = (int(mlp/mlp_range), int(turncnt/turncnt_range))

            if features == "MeanLateralPosition_SDSteeringAngle":
                cell = (int(mlp/mlp_range), int(stdsa/sdstd_range))


            sample.distance_to_target = us.manhattan(cell, goal)

            if sample.distance_to_target <= TARGET_THRESHOLD and misbehaviour == True:
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
        print(".", end='', flush=True)
        target_samples.append([sample.ind.m.sample_nodes, f"DeepHyperion", sample.misbehaviour])
    print("DeepHyperion", features, len(target_samples))
    return target_samples


def compute_distances_tSNE_vs_dh(df, div, approach):
    approach_data = []
    deephyperion_data = []
    
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        if row['y'] == f"{approach}-{div}":
            approach_data.append((row['tsne-2d-one'], row['tsne-2d-two']))
        if row['y'] == "DeepHyperion":
            deephyperion_data.append((row['tsne-2d-one'], row['tsne-2d-two']))


    distance_approach = compute_average_distance(approach_data)
    distance_deephyperion = compute_average_distance(deephyperion_data)


    list_data = [("DeepAtash", distance_approach, len(approach_data)),
                ("DeepHyperion", distance_deephyperion, len(deephyperion_data))]

    return  list_data
            
            
def compare_with_dh(approach, div, features, target_area):

    if div == "INPUT":
        threshold = 10.0


    result_folder = "../experiments/data/beamng"

    
    for feature in features:

        dst = f"../experiments/data/beamng/DeepAtash/{target_area}/{feature[0]}"
        dst_dh = f"../experiments/data/beamng/DeepHyperion/{feature[0]}"

        print(feature)
        for i in range(1, 6):
            for subdir, dirs, files in os.walk(dst_dh, followlinks=False):
                if "dh-"+str(i) in subdir and "beamng_nvidia_runner" in subdir:
                    inputs_dh = compute_targets_for_dh(subdir, feature[1], feature[0], threshold, div)
                    break
            
            inputs_focused = load_data(dst, str(i)+"-", approach, div)

            df = plot_tSNE(inputs_dh+inputs_focused, result_folder, feature[0], div)
            list_data = compute_distances_tSNE_vs_dh(df, div, approach)
            

            for data in list_data:
                dict_data = {
                    "approach": data[0],
                    "run": i,
                    "test input count": data[2],
                    "features": feature[0],
                    "euclidean sparseness": str(data[1][0]),
                    "manhattan sparseness": str(data[1][1])
                }

                filedest = f"{result_folder}/report_{data[0]}_{feature[0]}_{target_area}_{i}.json"
                with open(filedest, 'w') as f:
                    (json.dump(dict_data, f, sort_keys=True, indent=4))



if __name__ == "__main__": 

    dst = "../experiments/data/beamng/DeepAtash"
    feature_combinations = ["MeanLateralPosition_SegmentCount", "MeanLateralPosition_Curvature", "MeanLateralPosition_SDSteeringAngle"]
    find_best_div_approach(dst, feature_combinations)


    # goal cell for dark area  MLP-TurnCnt (160, 3), MLP-StdSA (162,108), Curv-StdSA (22, 75), MLP-Curv (167, 20)
    # feature and target
    # features = [("MeanLateralPosition_SDSteeringAngle",(162/mlp_range, 108/sdstd_range)), ("MeanLateralPosition_Curvature",(167/mlp_range, 20/curv_range)), ("MeanLateralPosition_SegmentCount", (160/mlp_range, 3/turncnt_range))]#("Curvature_SDSteeringAngle",(22/curv_range, 75/sdstd_range))]
    
  
    # compare_with_dh("nsga2", "INPUT", features, "target_cell_in_dark")