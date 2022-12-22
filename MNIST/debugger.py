    
import json
import os

feature_combinations = ["Moves-Bitmaps", "Orientation-Bitmaps", "Orientation-Moves"] # []
dst = f"../experiments/data/mnist/DeepAtash"
evaluation_area = ["target_cell_in_dark", "target_cell_in_gray", "target_cell_in_white"]

count = 0
total = 0

for evaluate in evaluation_area:  
    for features in feature_combinations:
        print(features)
        for i in range(1, 11):
            inputs = []
            for subdir, dirs, files in os.walk(dst, followlinks=False):
                if features in subdir and str(i)+"-" in subdir and evaluate in subdir and "nsga2" in subdir and "LATENT" in subdir:
                    data_folder = subdir   
                    for subdir, dirs, files in os.walk(data_folder, followlinks=False):
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

                                total += 1

                                if json_data["misbehaviour"] == False:
                                    print(json_path)
                                    count += 1
                    

print(count)
print(total)




count = 0
total = 0

# evaluation_area = [ "target_cell_in_dark"] # "target_cell_in_gray",
# dst = f"../experiments/data/beamng/DeepAtash"
# feature_combinations = ["MeanLateralPosition_SegmentCount", "MeanLateralPosition_Curvature", "MeanLateralPosition_SDSteeringAngle"]
   
# for evaluate in evaluation_area:
    
#     for features in feature_combinations:
#         for i in range(1, 6):
#             inputs = []
#             for subdir, dirs, files in os.walk(f"{dst}/{evaluate}/{features}", followlinks=False):
#                 if features in subdir and str(i)+"-" in subdir and evaluate in subdir and "10h/sim_" in subdir:
#                     data_folder = subdir


#                 for subdir, dirs, files in os.walk(dst, followlinks=False):
#                         # Consider only the files that match the pattern
#                         for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]: 

#                                 if "ga_" in json_path:
#                                     y2 = "GA"
#                                 elif "nsga2" in json_path:
#                                     y2 = "NSGA2"

#                                 y1 = "INPUT"
                        
#                                 with open(json_path) as jf:
#                                     json_data = json.load(jf)


#                                 total += 1
#                                 inputs.append([json_data["sample_nodes"], f"{y2}-{y1}", json_data['misbehaviour'], float(json_data["distance to target"])])
#                                 if json_data["misbehaviour"] == False:
#                                                 print(json_path)
#                                                 count += 1


# print(count)
# print(total)