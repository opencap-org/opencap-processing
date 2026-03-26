'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_kinetics.py
    ---------------------------------------------------------------------------
    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    This code makes use of CasADi, which is licensed under LGPL, Version 3.0;
    https://github.com/casadi/casadi/blob/master/LICENSE.txt.
    
    Install requirements:
        - Visit https://github.com/stanfordnmbl/opencap-processing for details.        
        - Third-party software packages:
            - CMake: https://cmake.org/download/.
            - (Windows only)
                - Visual studio: https://visualstudio.microsoft.com/downloads/.
                    - Make sure you install C++ support.
                    - Code tested with community editions 2017-2019-2022.
            
    Please contact us for any questions: https://www.opencap.ai/#contact
'''

# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys

baseDir = os.getcwd()
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(baseDir)
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

# %% User inputs.
'''
Please provide:
    
    session_id:     This is a 36 character-long string. You can find the ID of
                    all your sessions at https://app.opencap.ai/sessions.
                    
    trial_name:     This is the name of the trial you want to simulate. You can
                    find all trial names after loading a session.
                    
    motion_type:    This is the type of activity you want to simulate. Options
                    are 'running', 'walking', 'drop_jump', 'sit-to-stand', and
                    'squats'. We provide pre-defined settings that worked well
                    for this set of activities. If your activity is different,
                    select 'other' to use generic settings or set your own
                    settings in settingsOpenSimAD. See for example how we tuned
                    the 'running' settings to include periodic constraints in
                    the 'my_periodic_running' settings.
                    
    time_window:    This is the time interval you want to simulate. It is
                    recommended to simulate trials shorter than 2s. Set to []
                    to simulate full trial. For 'squats' or 'sit_to_stand', we
                    built segmenters to separate the different repetitions. In
                    such case, instead of providing the time_window, you can
                    provide the index of the repetition (see below) and the
                    time_window will be automatically computed.
                    
    repetition:     Only if motion_type is 'sit_to_stand' or 'squats'. This
                    is the index of the repetition you want to simulate (0 is 
                    first). There is no need to set the time_window. 
                    
    case:           This is a string that will be appended to the file names
                    of the results. Dynamic simulations are optimization
                    problems, and it is common to have to play with some
                    settings to get the problem to converge or converge to a
                    meaningful solution. It is useful to keep track of which
                    solution corresponds to which settings; you can then easily
                    compare results generated with different settings.
                    
    (optional)
    treadmill_speed:This an optional parameter that indicates the speed of
                    the treadmill in m/s. A positive value indicates that the
                    subject is moving forward. You should ignore this parameter
                    or set it to 0 if the trial was not measured on a
                    treadmill. By default, treadmill_speed is set to 0.
    (optional)
    contact_side:   This an optional parameter that indicates on which foot to
                    add contact spheres to model foot-ground contact. It might
                    be useful to only add contact spheres on one foot if only
                    that foot is in contact with the ground. We found this to
                    be helpful for simulating for instance single leg dropjump
                    as it might prevent the optimizer to cheat by using the
                    other foot to stabilize the model. Options are 'all', 
                    'left', and 'right'. By default, contact_side is set to
                    'all', meaning that contact spheres are added to both feet.
    
See example inputs below for different activities. Please note that we did not
verify the biomechanical validity of the results; we only made sure the
simulations converged to kinematic solutions that were visually reasonable.

Please contact us for any questions: https://www.opencap.ai/#contact
'''

# We provide a few examples for overground and treadmill activities.
# Select which example you would like to run.
session_type = 'overground' # Options are 'overground' and 'treadmill'.
session_id = "4d5c3eb1-1a59-4ea1-9178-d3634610561c"
# session_id = "e742eb1c-efbc-4c17-befc-a772150ca84d"
case = '0' # Change this to compare across settings.
# Options are 'squat', 'STS', and 'jump'.
if session_type == 'overground': 
    trial_name = 'squat'
    if trial_name == 'squat': # Squat
        motion_type = 'squats'
        repetition = 0
    elif trial_name == 'STS': # Sit-to-stand        
        motion_type = 'sit_to_stand'
        repetition = 1
    elif trial_name == 'jump': # Jump  
        motion_type = 'jumping'
        time_window = [1.3, 2.2]
    elif trial_name == 'SLDJ_L2': # Jump  
        motion_type = 'drop_jump_knee_adduction'
        time_window = [2.7, 3.7]
        contact_side = 'left'
# Options are 'walk_1_25ms', 'run_2_5ms', and 'run_4ms'.
elif session_type == 'treadmill': 
    trial_name = 'walk_1_25ms'
    torque_driven_model = False # Example with torque-driven model.
    if trial_name == 'walk_1_25ms': # Walking, 1.25 m/s
        motion_type = 'walking'
        time_window = [1.0, 2.5]
        treadmill_speed = 1.25
    elif trial_name == 'run_2_5ms': # Running, 2.5 m/s
        if torque_driven_model:
            motion_type = 'running_torque_driven'
        else:
            motion_type = 'running'
        time_window = [1.4, 2.6]
        treadmill_speed = 2.5
    elif trial_name == 'run_4ms': # Running with periodic constraints, 4.0 m/s
        motion_type = 'my_periodic_running'
        time_window = [3.1833333, 3.85]
        treadmill_speed = 4.0
    
# Set to True to solve the optimal control problem.
solveProblem = True
# Set to True to analyze the results of the optimal control problem. If you
# solved the problem already, and only want to analyze/process the results, you
# can set solveProblem to False and run this script with analyzeResults set to
# True. This is useful if you do additional post-processing but do not want to
# re-run the problem.
analyzeResults = True

# Path to where you want the data to be downloaded.
dataFolder = os.path.join(baseDir, 'Data')

# %% Setup. 
if not 'time_window' in locals():
    time_window = None
if not 'repetition' in locals():
    repetition = None
if not 'treadmill_speed' in locals():
    treadmill_speed = 0
if not 'contact_side' in locals():
    contact_side = 'all'
settings = processInputsOpenSimAD(baseDir, dataFolder, session_id, trial_name, 
                                  motion_type, time_window, repetition,
                                  treadmill_speed, contact_side)

if case == '1':
    settings['meshDensity'] = 100

# %% Simulation.
run_tracking(baseDir, dataFolder, session_id, settings, case=case, 
              solveProblem=solveProblem, analyzeResults=analyzeResults)

# %% Plots.
# To compare different cases, add to the cases list, eg cases=['0','1'].
plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, cases=[case], mainPlots=False)
