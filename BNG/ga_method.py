from datetime import datetime
import json
import sys
import random
from deap import base, creator, tools
import logging as log
from pathlib import Path
import numpy as np

# local
from core.config import Config
import self_driving.beamng_config as cfg
import self_driving.beamng_individual as BeamNGIndividual
from archive import Archive
from config import POPSIZE, to_json
import self_driving.beamng_problem as BeamNGProblem
from config import GOAL, FEATURES, POPSIZE, RESEEDUPPERBOUND,\
     RUN_TIME, DIVERSITY_METRIC, META_FILE, TARGET_SIZE
import utils as us
from sample import Sample
from evaluator import Evaluator
from feature import Feature

evaluator = Evaluator()

# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Define the individual.
creator.create("Individual", Sample, fitness=creator.FitnessMin)



def generate_initial_pop(problem):
    samples = []
    initialpop = 48
    for i in range(1, initialpop+1):
        max_angle = random.randint(10,100)
        road = problem.generate_random_member(max_angle)
        ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
        ind.seed = i
        sample = creator.Individual(ind)
        samples.append(sample)        
    return samples 


def reseed_individual(problem, seed):
    max_angle = random.randint(10,100)
    road = problem.generate_random_member(max_angle)
    ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
    ind.oob_ff = None
    ind.seed = seed
    sample = creator.Individual(ind)
    return sample


def evaluate_individual(individual, features, goal, archive):

    if individual.ind.oob_ff == None:
        evaluator.evaluate(individual.ind)

    # diversity computation
    individual.sparseness, _ = evaluator.evaluate_sparseness(individual, archive.archive)

    if individual.distance_to_target == None:         
        # original coordinates
        b = tuple()

        individual.features = {
                    "segment_count": us.segment_count(individual),
                    "sd_steering_angle": us.sd_steering(individual),
                    "mean_lateral_position": us.mean_lateral_position(individual),
                    "curvature": us.curvature(individual) 
        }
        
        for ft in features:
            i = ft.get_coordinate_for(individual)
            if i != None:
                b = b + (i,)
            else:
                # this individual is out of range and must be discarded
                individual.distance_to_target = np.inf
                return (np.inf, )
        
        individual.coordinate = b
        individual.distance_to_target = us.manhattan(b, goal)
    
    return (individual.distance_to_target,)

def mutate_individual(individual):
    sample = individual.clone()
    sample.ind.mutate()
    return sample

def generate_features(meta_file):

    features = []
    with open(meta_file, 'r') as f:
        meta = json.load(f)["features"]
        
    if "MeanLateralPosition" in FEATURES:
        f3 = Feature("mean_lateral_position", meta["mean_lateral_position"]["min"], meta["mean_lateral_position"]["max"], "mean_lateral_position", 25)
        features.append(f3)
    if "Curvature" in FEATURES:
        f1 = Feature("curvature",meta["curvature"]["min"], meta["curvature"]["max"], "curvature", 25)
        features.append(f1)
    if "SegmentCount" in FEATURES:
        f2 = Feature("segment_count",meta["segment_count"]["min"], meta["segment_count"]["max"], "segment_count", 8)
        features.append(f2)
    if "SDSteeringAngle" in FEATURES:
        f1 = Feature("sd_steering_angle",meta["sd_steering_angle"]["min"], meta["sd_steering_angle"]["max"], "sd_steering", 25)
        features.append(f1)
    return features
       
def goal_acheived(population):
    for sample in population:
        if sample.distance_to_target == 0:
            print("Goal achieved")
            return True
    return False

toolbox.register("population", generate_initial_pop)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selBest)
toolbox.register("mutate", mutate_individual)


def main(name):
    # initialize archive
    archive = Archive(TARGET_SIZE)

    last_seed_index = POPSIZE
    _config = cfg.BeamNGConfig()
    _config.name = name
    problem = BeamNGProblem.BeamNGProblem(_config)

    start_time = datetime.now()

    features = generate_features(META_FILE)


    # Generate initial population and feature map 
    log.info(f"Generate initial population")  
           

    population = toolbox.population(problem)

    # Evaluate the individuals
    invalid_ind = [ind for ind in population]

    fitnesses = [toolbox.evaluate(i, features, GOAL, archive) for i in invalid_ind]

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Select the next generation population
    population = toolbox.select(population, POPSIZE)


    # Begin the generational process
    gen = 1

    condition = True

    while condition:
        log.info(f"Iteration: {gen}")

        # Vary the population.
        offspring = []
        for ind in population:
            sample = creator.Individual(ind.ind)
            sample.ind.oob_ff = None
            offspring.append(sample)
                
        # Mutation.
        log.info("Mutation")
        for ind in offspring:
            toolbox.mutate(ind)

        # Reseeding
        if len(archive.archive) > 0:
            log.info(f"Reseeding")
            seed_range = random.randrange(1, RESEEDUPPERBOUND)
            candidate_seeds = archive.archived_seeds
            log.info(f"Archive seeds: {candidate_seeds}")

            for i in range(len(population)):
                if population[i].seed in archive.archived_seeds:
                    ind = population[i]
                    new_ind = reseed_individual(problem, last_seed_index)
                    population[i] = new_ind
                    log.info(f"ind {ind.id} with seed {ind.seed} reseeded by {new_ind.id} with seed {new_ind.seed}")
                    last_seed_index += 1


        # Evaluate the individuals
        invalid_ind = [ind for ind in offspring + population]

        fitnesses = [toolbox.evaluate(i, features, GOAL, archive) for i in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Update archive
        for ind in offspring + population:
            is_added = archive.update_archive(ind, evaluator)
            # if a new ind added to archive, recompute all the sparsenesses
            if is_added:
                archive.recompute_sparseness_archive(evaluator)

        # Select the next generation population
        population = toolbox.select(population + offspring, POPSIZE)

        for individual in population:
            log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['segment_count']}, {individual.features['curvature']}, {individual.features['sd_steering_angle']}, {individual.features['mean_lateral_position']}), performance {individual.ind.oob_ff} and distance {individual.distance_to_target} selected")



        gen += 1
        log.info(f"Archive: {len(archive.archive)}")
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time


        # if Config.EXECTIME > 10800 and h_3:
        #     archive.export_archive(name+"/3h")
        #     h_3 = False
        # elif Config.EXECTIME > 18000 and h_5:
        #     archive.export_archive(name+"/5h")
        #     h_5 = False
        if Config.EXECTIME > RUN_TIME: #gen == GEN: #or goal_acheived(population):
            condition = False

        log.info("Elapsed time: " + str(elapsed_time))
        log.info("EXECTIME: " + str(Config.EXECTIME))
        log.info(f"Archive: {len(archive.archive)}")

    archive.export_archive(name)
    return population


if __name__ == "__main__": 

    start_time = datetime.now()
    run = sys.argv[1]
    name = f"logs/{run}-ga_-features_{FEATURES[0]}-{FEATURES[1]}-diversity_{DIVERSITY_METRIC}"
    
    Path(name).mkdir(parents=True, exist_ok=True)

    to_json(name)
    log_to = f"{name}/logs.txt"

    # Setup logging
    us.setup_logging(log_to)
    print("Logging results to " + log_to)

    pop = main(name)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    log.info("elapsed_time:"+ str(elapsed_time))

    print("GAME OVER")

    

    
