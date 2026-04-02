'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_sts_analysis.py
    ---------------------------------------------------------------------------
    Copyright TBD

    Author(s): Scott Uhlrich, RD Magruder

    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run a kinematic analysis of sit-to-stand data. It also
    includes optional inverse dynamics, and you can compute scalar metrics.

'''


import os
import sys
sys.path.append("../")
sys.path.append("../ActivityAnalyses")
from ActivityAnalyses.sts_analysis import sts_analysis
from utils import get_trial_id, download_trial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

baseDir = os.path.join(os.getcwd(), '..')

# %% Paths.
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
session_id = 'f6d66b1a-ce0b-4c38-b6ef-eb48e8a4b49d'
# trial_name = 'sts_normal' # Normal sit-to-stand; repetitions 3 and 4 are good for this trial
trial_name = 'sts_lean' # Lean forward sit-to-stand
# trial_name = 'sts_lean2' # Lean forward sit-to-stand
# trial_name = 'sts_momentum' # Sit-to-stand with momentum

# Select how many sit-to-stand cycles you'd like to analyze. Select -1 for all
# sit-to-stand cycles detected in the trial.
n_sts_cycles = -1

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# Select whether you want to run inverse dynamics, and select settings.
run_inverse_dynamics = False
case ='0' # Change this to compare across settings.
motion_type = 'sit_to_stand'
repetition = -1 # Select -1 for all repetitions

scalar_names = {
    'rise_time', 'sts_time', # Commonly reported metrics for sit-to-stand
    'torso_orientation', 'torso_angular_velocity', # Torso kinematics PRIOR to liftoff; using world frame
    'torso_orientation_liftoff', 'torso_angular_velocity_liftoff', # Torso kinematics AT liftoff; using world frame
}

# %% Download trial.
trial_id = get_trial_id(session_id, trial_name)

sessionDir = os.path.join(dataFolder, session_id)

trialName = download_trial(trial_id, sessionDir, session_id=session_id)

# %% Run sit-to-stand analysis.
sts = sts_analysis(sessionDir, trialName, n_sts_cycles=n_sts_cycles,
                   lowpass_cutoff_frequency_for_coordinate_values=filter_frequency)

if run_inverse_dynamics:
    sts_kinetics = sts.run_dynamic_simulation(baseDir, dataFolder, session_id, trial_name, case=case, repetition=repetition, verbose=True)

# %% Plot sit-to-stand events.
time = sts.coordinateValues['time']
pelvis_Sig = sts.coordinateValues['pelvis_ty']
torso_z = sts.body_angles['torso_z']
velTorso_z = sts.body_angular_velocity['torso_z']

plt.figure()
plt.plot(time, pelvis_Sig)
for c_v, val in enumerate(sts.stsEvents['endRisingIdx']):
    plt.plot(time[val], pelvis_Sig[val], marker='o', markerfacecolor='k',
             markeredgecolor='none', linestyle='none',
             label='End Rising')
    val2 = sts.stsEvents['startRisingIdx'][c_v]
    plt.plot(time[val2], pelvis_Sig[val2], marker='o',
             markerfacecolor='r', markeredgecolor='none',
             linestyle='none', label='Rising start')
    val3 = sts.stsEvents['sittingIdx'][c_v]
    plt.plot(time[val3], pelvis_Sig[val3], marker='o',
             markerfacecolor='g', markeredgecolor='none',
             linestyle='none',
             label='Sitting Time')
    val4 = sts.stsEvents['forwardLeanIdx'][c_v]
    plt.plot(time[val4], pelvis_Sig[val4], marker='o',
             markerfacecolor='b', markeredgecolor='none',
             linestyle='none', label='Forward Lean Start')
    if c_v == 0:
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.40), ncol=2)
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Vertical pelvis position during sit-to-stand')
ticks = np.arange(np.floor(np.min(time)), np.ceil(np.max(time)) + 1, 2)
plt.xticks(ticks)
plt.tight_layout()
plt.show()

# %% Display scalars of interest.
stsResults = {'scalars': sts.compute_scalars(scalar_names)}
print()
print("Average STS Results:")
for key, value in sorted(stsResults['scalars'].items()):
    # Check if the value is a tuple of 2 values (e.g., left and right)
    if isinstance(value['value'], tuple):
        left_value, right_value = value['value']
        left_value_rounded = round(left_value, 2)
        right_value_rounded = round(right_value, 2)
        print(f"{key}: Left = {left_value_rounded} {value['units']}, Right = {right_value_rounded} {value['units']}")
    else:
        rounded_value = round(value['value'], 2)
        print(f"{key}: {rounded_value} {value['units']}")

print()
print("STS Results per cycle:")
stsResults_cycles = sts.compute_scalars(scalar_names, return_all=True)
for key, values in sorted(stsResults_cycles.items()):
    rounded_values = [round(value, 4) for value in values['value']]
    print(f"{key} ({values['units']}): {rounded_values}")

