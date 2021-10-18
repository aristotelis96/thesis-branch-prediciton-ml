import csv
import glob
import multiprocessing
import os
import re
import subprocess
import yaml

SIMPOINT_LENGTH = 100000000

__env_dir__ = os.path.dirname(__file__) + '/../environment_setup'
__paths_file__ = __env_dir__ + '/paths.yaml'

assert os.path.exists(__paths_file__), (
  ('Expecting a paths.yaml file at {}. You have to create one following '
   'the format of paths_example.yaml in the same directory').format(__paths_file__))

with open(__paths_file__) as f:
  PATHS = yaml.safe_load(f)

def get_brs_from_accuracy_file(path_to_file):
    brs = []
    with open(path_to_file, 'r') as fptr:
        for line in fptr:
            if "Printing stats for each H2P" in line: break
        for line in fptr:
            h2p = int(line.split(" ")[0])
            brs.append(h2p)
    return brs

def read_hard_brs_from_accuracy_files(benchmark, predictor='TAGE8'):
    # Find common H2P among all files    
    # Read files except ref
    measure_accuracy_files = filter(
        lambda x: "ref" not in x,
        os.listdir('{}/{}/{}'.format(
            PATHS['measure_H2P_dir'], predictor, benchmark)))
    # Initialize result
    brs_in_files = []
    for measure_accuracy_file in measure_accuracy_files:
        path_to_file = '{}/{}/{}/{}'.format(
            PATHS['measure_H2P_dir'], predictor, benchmark, measure_accuracy_file)
        brs_in_files.append(set( get_brs_from_accuracy_file(path_to_file)))
    # Find H2Ps that intersect in 3 or more files
    brs = set()
    allBrs = set.union(*brs_in_files)
    for br in allBrs:
        occurances = 0
        for brs_in_file in brs_in_files:
            if br in brs_in_file:
                occurances+=1
            if(occurances>=len([x for x in os.listdir("{}/{}/{}".format(PATHS['measure_H2P_dir'], predictor, benchmark)) if "ref" not in x])*0.4):
                brs.add(br)
                continue
    return sorted(list(brs))

def read_hard_brs(benchmark, name):
    filepath = '{}/{}_{}'.format( PATHS['hard_brs_dir'], benchmark, name)
    with open (filepath) as f:
        return [int(x,16) for x in f.read().splitlines()]

