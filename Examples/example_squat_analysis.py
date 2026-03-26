'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_squat_analysis.py
    ---------------------------------------------------------------------------
    Copyright 2024 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Carmichael Ong
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
                
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run a kinematic analysis of squat data.
    
'''

import os
import sys
sys.path.append("../")
sys.path.append("../ActivityAnalyses")
import numpy as np

from squat_analysis import squat_analysis
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe_with_shading

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.

# example without KA
session_id = '1b380cb9-3c5c-498c-bc30-a1a379ffa04c'
#trial_name = 'bossu_squats'
trial_name = 'single_leg_squats_l'

# example with KA
#session_id = 'e742eb1c-efbc-4c17-befc-a772150ca84d'
#trial_name = 'SLS_L1'

# Select how many repetitions you'd like to analyze. Select -1 for all
# repetitions detected in the trial.
n_repetitions = -1

# Select lowpass filter frequency for kinematics data.
filter_frequency = 4

# %% Gait analysis.
# Get trial id from name.
trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

# Download data.
trialName = download_trial(trial_id,sessionDir,session_id=session_id) 

# Init gait analysis.
squat = squat_analysis(
    sessionDir, trialName,
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    n_repetitions=n_repetitions)

# Detect squat type
squat_events = squat.get_squat_events()
eventTypes = squat.squatEvents['eventTypes']
print('Detected {} total squats: {} single_leg_l, {} single_leg_r, {} double_leg.'.format( 
      len(eventTypes), eventTypes.count('single_leg_l'), 
      eventTypes.count('single_leg_r'), eventTypes.count('double_leg')))
print('')

unique_types = set(eventTypes)
if len(unique_types) < 1:
    raise Exception('No squats detected.')
if len(unique_types) > 1:
    raise Exception('More than one type of squat detected.')

# Check if we can also analyze adduction
knee_adduction_names = ['knee_adduction_r', 'knee_adduction_l']
analyze_knee_adduction = False
if all(x in squat.coordinates for x in knee_adduction_names):
    analyze_knee_adduction = True

# Example metrics
max_trunk_lean_ground_mean, max_trunk_lean_ground_std, max_trunk_lean_ground_units = squat.compute_trunk_lean_relative_to_ground()
max_trunk_flexion_mean, max_trunk_flexion_std, max_trunk_flexion_units = squat.compute_trunk_flexion_relative_to_ground()
squat_depth_mean, squat_depth_std, squat_depth_units = squat.compute_squat_depth()

print('Squat metrics summary:')
print('Max trunk lean relative to ground: {} +/- {} {}'.format(np.round(max_trunk_lean_ground_mean,2), np.round(max_trunk_lean_ground_std,2), max_trunk_lean_ground_units))
print('Max trunk flexion flexion to ground: {} +/- {} {}'.format(np.round(max_trunk_flexion_mean,2), np.round(max_trunk_flexion_std,2), max_trunk_flexion_units))
print('Squat depth: {} +/- {} {}'.format(np.round(squat_depth_mean,2), np.round(squat_depth_std,2), squat_depth_units))
print('')

if eventTypes[0] == 'double_leg':
    ratio_max_knee_flexion_angle_mean, ratio_max_knee_flexion_angle_std, ratio_max_knee_flexion_angle_unit = squat.compute_ratio_peak_angle('knee_angle_r', 'knee_angle_l')
    ratio_max_hip_flexion_angle_mean, ratio_max_hip_flexion_angle_std, ratio_max_knee_flexion_angle_unit = squat.compute_ratio_peak_angle('hip_flexion_r', 'hip_flexion_l')
   

if eventTypes[0] == 'double_leg' or eventTypes[0] == 'single_leg_r':
    max_hip_flexion_angle_r_mean, max_hip_flexion_angle_r_std, max_hip_flexion_angle_r_units = squat.compute_peak_angle('hip_flexion_r')
    max_knee_flexion_angle_r_mean, max_knee_flexion_angle_r_std, max_knee_flexion_angle_r_units = squat.compute_peak_angle('knee_angle_r')
    max_ankle_flexion_angle_r_mean, max_ankle_flexion_angle_r_std, max_ankle_flexion_angle_r_units = squat.compute_peak_angle('ankle_angle_r')
    max_hip_adduction_angle_r_mean, max_hip_adduction_angle_r_std, max_hip_adduction_angle_r_units = squat.compute_peak_angle('hip_adduction_r')
    rom_knee_flexion_angle_r_mean, rom_knee_flexion_angle_r_std, rom_knee_flexion_angle_r_units = squat.compute_range_of_motion('knee_angle_r')

    print('Right side summary:')
    print('Peak hip flexion angle: {} +/- {} {}'.format(np.round(max_hip_flexion_angle_r_mean,2), np.round(max_hip_flexion_angle_r_std,2), max_hip_flexion_angle_r_units))
    print('Peak knee flexion angle: {} +/- {} {}'.format(np.round(max_knee_flexion_angle_r_mean,2), np.round(max_knee_flexion_angle_r_std,2), max_knee_flexion_angle_r_units))
    print('Peak ankle dorsiflexion angle: {} +/- {} {}'.format(np.round(max_ankle_flexion_angle_r_mean,2), np.round(max_ankle_flexion_angle_r_std,2), max_ankle_flexion_angle_r_units)) 
    
    if analyze_knee_adduction:
        max_knee_adduction_angle_r_mean, max_knee_adduction_angle_r_std, max_knee_adduction_angle_r_units = squat.compute_peak_angle('knee_adduction_r')
        print('Peak knee adduction angle: {} +/- {} {}'.format(np.round(max_knee_adduction_angle_r_mean,2), np.round(max_knee_adduction_angle_r_std,2), max_knee_adduction_angle_r_units))
        
    print('')
    
if eventTypes[0] == 'double_leg' or eventTypes[0] == 'single_leg_l':
    max_hip_flexion_angle_l_mean, max_hip_flexion_angle_l_std, max_hip_flexion_angle_l_units = squat.compute_peak_angle('hip_flexion_l')
    max_knee_flexion_angle_l_mean, max_knee_flexion_angle_l_std, max_knee_flexion_angle_l_units = squat.compute_peak_angle('knee_angle_l')
    max_ankle_flexion_angle_l_mean, max_ankle_flexion_angle_l_std, max_ankle_flexion_angle_l_units = squat.compute_peak_angle('ankle_angle_l')
    max_hip_adduction_angle_l_mean, max_hip_adduction_angle_l_std, max_hip_adduction_angle_l_units = squat.compute_peak_angle('hip_adduction_l')
    rom_knee_flexion_angle_l_mean, rom_knee_flexion_angle_l_std, rom_knee_flexion_angle_l_units = squat.compute_range_of_motion('knee_angle_l')

    print('Left side summary:')
    print('Peak hip flexion angle: {} +/- {} {}'.format(np.round(max_hip_flexion_angle_l_mean,2), np.round(max_hip_flexion_angle_l_std,2), max_hip_flexion_angle_l_units))
    print('Peak knee flexion angle: {} +/- {} {}'.format(np.round(max_knee_flexion_angle_l_mean,2), np.round(max_knee_flexion_angle_l_std,2), max_knee_flexion_angle_l_units))
    print('Peak ankle dorsiflexion angle: {} +/- {} {}'.format(np.round(max_ankle_flexion_angle_l_mean,2), np.round(max_ankle_flexion_angle_l_std,2), max_ankle_flexion_angle_l_units))
    
    if analyze_knee_adduction:
        max_knee_adduction_angle_l_mean, max_knee_adduction_angle_l_std, max_knee_adduction_angle_l_units = squat.compute_peak_angle('knee_adduction_l')
        print('Peak knee adduction angle: {} +/- {} {}'.format(np.round(max_knee_adduction_angle_l_mean,2), np.round(max_knee_adduction_angle_l_std,2), max_knee_adduction_angle_l_units))
        
    print('')

# Example metrics to json: aggregated over both legs
max_knee_flexion_angle_r_mean, max_knee_flexion_angle_r_std, _ = squat.compute_peak_angle('knee_angle_r')
max_knee_flexion_angle_l_mean, max_knee_flexion_angle_l_std, _ = squat.compute_peak_angle('knee_angle_l')
max_knee_flexion_angle_mean_mean = np.mean(np.array([max_knee_flexion_angle_r_mean, max_knee_flexion_angle_l_mean]))
max_knee_flexion_angle_mean_std = np.mean(np.array([max_knee_flexion_angle_r_std, max_knee_flexion_angle_l_std]))

max_hip_flexion_angle_r_mean, max_hip_flexion_angle_r_std, _ = squat.compute_peak_angle('hip_flexion_r')
max_hip_flexion_angle_l_mean, max_hip_flexion_angle_l_std, _ = squat.compute_peak_angle('hip_flexion_l')
max_hip_flexion_angle_mean_mean = np.mean(np.array([max_hip_flexion_angle_r_mean, max_hip_flexion_angle_l_mean]))
max_hip_flexion_angle_mean_std = np.mean(np.array([max_hip_flexion_angle_r_std, max_hip_flexion_angle_l_std]))

max_hip_adduction_angle_r_mean, max_hip_adduction_angle_r_std, _ = squat.compute_peak_angle('hip_adduction_r')
max_hip_adduction_angle_l_mean, max_hip_adduction_angle_l_std, _ = squat.compute_peak_angle('hip_adduction_l')
max_hip_adduction_angle_mean_mean = np.mean(np.array([max_hip_adduction_angle_r_mean, max_hip_adduction_angle_l_mean]))
max_hip_adduction_angle_mean_std = np.mean(np.array([max_hip_adduction_angle_r_std, max_hip_adduction_angle_l_std]))

rom_knee_flexion_angle_r_mean, rom_knee_flexion_angle_r_std, _ = squat.compute_range_of_motion('knee_angle_r')
rom_knee_flexion_angle_l_mean, rom_knee_flexion_angle_l_std, _ = squat.compute_range_of_motion('knee_angle_l')
rom_knee_flexion_angle_mean_mean = np.mean(np.array([rom_knee_flexion_angle_r_mean, rom_knee_flexion_angle_l_mean]))
rom_knee_flexion_angle_mean_std = np.mean(np.array([rom_knee_flexion_angle_r_std, rom_knee_flexion_angle_l_std]))

print('Aggregating over both legs:')
print('Peak knee flexion angle: {} +/- {} deg'.format(np.round(max_knee_flexion_angle_mean_mean,2), np.round(max_knee_flexion_angle_mean_std,2)))
print('Peak hip flexion angle: {} +/- {} deg'.format(np.round(max_hip_flexion_angle_mean_mean,2), np.round(max_hip_flexion_angle_mean_std,2)))
print('Peak hip adduction angle: {} +/- {} deg'.format(np.round(max_hip_adduction_angle_mean_mean,2), np.round(max_hip_adduction_angle_mean_std,2)))
print('ROM knee flexion angle: {} +/- {} deg'.format(np.round(rom_knee_flexion_angle_mean_mean,2), np.round(rom_knee_flexion_angle_mean_std,2)))

squat_scalars = {}
squat_scalars['peak_knee_flexion_angle_mean'] = {'value': max_knee_flexion_angle_mean_mean}
squat_scalars['peak_knee_flexion_angle_mean']['label'] = 'Mean peak knee flexion angle (deg)'
squat_scalars['peak_knee_flexion_angle_mean']['colors'] = ["red", "yellow", "green"]
peak_knee_flexion_angle_threshold = 100
squat_scalars['peak_knee_flexion_angle_mean']['min_limit'] = float(np.round(0.90*peak_knee_flexion_angle_threshold))
squat_scalars['peak_knee_flexion_angle_mean']['max_limit'] = float(peak_knee_flexion_angle_threshold)

squat_scalars['peak_knee_flexion_angle_std'] = {'value': max_knee_flexion_angle_mean_std}
squat_scalars['peak_knee_flexion_angle_std']['label'] = 'Std peak knee flexion angle (deg)'
squat_scalars['peak_knee_flexion_angle_std']['colors'] = ["green", "yellow", "red"]
std_threshold_min = 2
std_threshold_max = 4
squat_scalars['peak_knee_flexion_angle_std']['min_limit'] = float(std_threshold_min)
squat_scalars['peak_knee_flexion_angle_std']['max_limit'] = float(std_threshold_max)

squat_scalars['peak_hip_flexion_angle_mean'] = {'value': max_hip_flexion_angle_mean_mean}
squat_scalars['peak_hip_flexion_angle_mean']['label'] = 'Mean peak hip flexion angle (deg)'
squat_scalars['peak_hip_flexion_angle_mean']['colors'] = ["red", "yellow", "green"]
peak_hip_flexion_angle_threshold = 100
squat_scalars['peak_hip_flexion_angle_mean']['min_limit'] = float(np.round(0.90*peak_hip_flexion_angle_threshold))
squat_scalars['peak_hip_flexion_angle_mean']['max_limit'] = float(peak_hip_flexion_angle_threshold)

squat_scalars['peak_hip_flexion_angle_std'] = {'value': max_hip_flexion_angle_mean_std}
squat_scalars['peak_hip_flexion_angle_std']['label'] = 'Std peak hip flexion angle (deg)'
squat_scalars['peak_hip_flexion_angle_std']['colors'] = ["green", "yellow", "red"]
squat_scalars['peak_hip_flexion_angle_std']['min_limit'] = float(std_threshold_min)
squat_scalars['peak_hip_flexion_angle_std']['max_limit'] = float(std_threshold_max)

squat_scalars['peak_knee_adduction_angle_mean'] = {'value': max_hip_adduction_angle_mean_mean}
squat_scalars['peak_knee_adduction_angle_mean']['label'] = 'Mean peak knee adduction angle (deg)'
squat_scalars['peak_knee_adduction_angle_mean']['colors'] = ["red", "green", "red"]
knee_adduction_angle_threshold = 5
squat_scalars['peak_knee_adduction_angle_mean']['min_limit'] = float(-knee_adduction_angle_threshold)
squat_scalars['peak_knee_adduction_angle_mean']['max_limit'] = float(knee_adduction_angle_threshold)

squat_scalars['peak_knee_adduction_angle_std'] = {'value': max_hip_adduction_angle_mean_std}
squat_scalars['peak_knee_adduction_angle_std']['label'] = 'Std peak knee adduction angle (deg)'
squat_scalars['peak_knee_adduction_angle_std']['colors'] = ["green", "yellow", "red"]
squat_scalars['peak_knee_adduction_angle_std']['min_limit'] = float(std_threshold_min)
squat_scalars['peak_knee_adduction_angle_std']['max_limit'] = float(std_threshold_max)

squat_scalars['rom_knee_flexion_angle_mean'] = {'value': rom_knee_flexion_angle_mean_mean}
squat_scalars['rom_knee_flexion_angle_mean']['label'] = 'Mean range of motion knee flexion angle (deg)'
squat_scalars['rom_knee_flexion_angle_mean']['colors'] = ["red", "yellow", "green"]
rom_knee_flexion_angle_threshold_min = 85
rom_knee_flexion_angle_threshold_max = 115
squat_scalars['rom_knee_flexion_angle_mean']['min_limit'] = float(rom_knee_flexion_angle_threshold_min)
squat_scalars['rom_knee_flexion_angle_mean']['max_limit'] = float(rom_knee_flexion_angle_threshold_max)

squat_scalars['rom_knee_flexion_angle_std'] = {'value': rom_knee_flexion_angle_mean_std}
squat_scalars['rom_knee_flexion_angle_std']['label'] = 'Std range of motion knee flexion angle (deg)'
squat_scalars['rom_knee_flexion_angle_std']['colors'] = ["green", "yellow", "red"]
squat_scalars['rom_knee_flexion_angle_std']['min_limit'] = float(std_threshold_min)
squat_scalars['rom_knee_flexion_angle_std']['max_limit'] = float(std_threshold_max)

# dump to json
import json
with open('squat_scalars.json', 'w') as fp:
    json.dump(squat_scalars, fp)

    
# %% Example plots.
squat_joint_kinematics = squat.get_coordinates_segmented_normalized_time()
squat_com_kinematics = squat.get_center_of_mass_segmented_normalized_time()

plot_dataframe_with_shading(
    [squat_joint_kinematics['mean']],
    [squat_joint_kinematics['sd']],
    xlabel = '% repetition',
    title = 'Joint kinematics (m or deg)')

plot_dataframe_with_shading(
            [squat_com_kinematics['mean']],
            [squat_com_kinematics['sd']],
            xlabel = '% repetition',
            title = 'center of mass kinematics (m)')