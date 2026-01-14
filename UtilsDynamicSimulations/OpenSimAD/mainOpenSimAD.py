'''
    ---------------------------------------------------------------------------
    OpenCap processing: mainOpenSimAD.py
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

    This script formulates and solves the trajectory optimization problem 
    underlying the tracking simulation of coordinates.
'''

# %% Packages.
import os
import casadi as ca
import numpy as np
import sys
import yaml
import scipy.interpolate as interpolate
import platform
import copy
import pandas as pd

# %% Settings.
def run_tracking(baseDir, dataDir, subject, settings, case='0',
                 solveProblem=True, analyzeResults=True, writeGUI=True,
                 computeKAM=True, computeMCF=True):
    
    # %% Settings.
    # Most available settings are left from trying out different formulations 
    # when solving the trajectory optimization problems. We decided to keep
    # them in this script in case users want to play with them or use them as
    # examples for creating their own formulations.    
    
    # Cost function weights.
    weights = {
        'jointAccelerationTerm': settings['weights']['jointAccelerationTerm'],
        'armExcitationTerm': settings['weights']['armExcitationTerm'],
        'positionTrackingTerm': settings['weights']['positionTrackingTerm'],
        'velocityTrackingTerm': settings['weights']['velocityTrackingTerm']}    
    if 'activationTerm' in settings['weights']:
        weights['activationTerm'] = settings['weights']['activationTerm']
    if 'forceDtTerm' in settings['weights']:
        weights['forceDtTerm'] = settings['weights']['forceDtTerm']
    if 'activationDtTerm' in settings['weights']:
        weights['activationDtTerm'] = settings['weights']['activationDtTerm']
    
    # Model info.
    # Model name.
    OpenSimModel = 'LaiUhlrich2022'
    if 'OpenSimModel' in settings:  
        OpenSimModel = settings['OpenSimModel']
    model_full_name = OpenSimModel + "_scaled_adjusted"
    
    # Set withMTP to True to include metatarshophalangeal joints.
    withMTP = True
    if 'withMTP' in settings:  
        withMTP = settings['withMTP']
    
    # Set withArms to True to include arm joints.
    withArms = True
    coordinate_optimal_forces = {}
    if "withArms" in settings:
         withArms = settings['withArms']
    
    # Set withLumbarCoordinateActuators to True to actuate the lumbar 
    # coordinates with coordinate actuators. Coordinate actuator have simple
    # dynamics to relate excitations and activations. The maximal excitations
    # and activations are given in boundsOpenSimAD. The excitations of the
    # coordinate actuators are minimized in the cost function with a weight
    # given by weights['lumbarExcitationTerm'].
    withLumbarCoordinateActuators = True
    if "withLumbarCoordinateActuators" in settings:
         withLumbarCoordinateActuators = (
             settings['withLumbarCoordinateActuators'])
    if withLumbarCoordinateActuators:   
         weights['lumbarExcitationTerm'] = 1
         if 'lumbarExcitationTerm' in settings['weights']:
            weights['lumbarExcitationTerm'] = (
                settings['weights']['lumbarExcitationTerm'])
    
    # Set scaleIsometricMuscleForce to scale the maximal isometric muscle
    # forces. By default, scaleIsometricMuscleForce is set to 1, and the forces
    # therefore correspond to those of the generic models.
    scaleIsometricMuscleForce = 1
    if 'scaleIsometricMuscleForce' in settings: 
        scaleIsometricMuscleForce = settings['scaleIsometricMuscleForce']
    
    # Set withReserveActuators to True to include reserve actuators. Reserve
    # actuators will be added for coordinates specified in
    # reserveActuatorCoordinates. Eg, 
    # 'reserveActuatorCoordinates'['hip_rotation_l'] = 30 will add a reserve
    # actuator to the hip_rotation_l with a maximal value of 30. The reserve
    # actuators are minimized in the cost function with a weight given by
    # weights['reserveActuatorTerm']
    withReserveActuators = False
    if 'withReserveActuators' in settings: 
        withReserveActuators = settings['withReserveActuators']
    if withReserveActuators:
        reserveActuatorCoordinates = settings['reserveActuatorCoordinates']
        weights['reserveActuatorTerm'] = 0.001
        if 'reserveActuatorTerm' in settings['weights']:
            weights['reserveActuatorTerm'] = (
                settings['weights']['reserveActuatorTerm'])
    
    # Set ignorePassiveFiberForce to True to ignore passive muscle forces.
    ignorePassiveFiberForce = False
    if 'ignorePassiveFiberForce' in settings: 
        ignorePassiveFiberForce = settings['ignorePassiveFiberForce']
     
    # lb_activation determines the lower bound on muscle activations.
    lb_activation = 0.01
    if 'lb_activation' in settings:
        lb_activation = settings['lb_activation']
        
    # Trials info.
    # Trial name.
    trialName = settings['trial_name']
    
    # Time interval.
    timeIntervals = settings['timeInterval']
    timeElapsed = timeIntervals[1] - timeIntervals[0]
    
    # Mesh intervals / density.
    if 'N' in settings: # number of mesh intervals.
        N = settings['N']
    else:
        meshDensity = 100 # default is N=100 for t=1s
        if 'meshDensity' in settings:
            meshDensity = settings['meshDensity']
        N = int(round(timeElapsed * meshDensity, 2))
        
    # Discretized time interval.
    tgrid = np.linspace(timeIntervals[0], timeIntervals[1], N+1)
    tgridf = np.zeros((1, N+1))
    tgridf[:,:] = tgrid.T            
        
    # If heel_vGRF_threshold is larger than 0, a constraint will enforce
    # that the contact spheres on the model's heels generate a vertical
    # ground reaction force larger than heel_vGRF_threshold. This is used
    # to force the model to keep its heels in contact with the ground.
    heel_vGRF_threshold = 0
    if 'heel_vGRF_threshold' in settings:
        heel_vGRF_threshold = settings['heel_vGRF_threshold']
        
    # If min_ratio_vGRF is True and weights['vGRFRatioTerm'] is 
    # larger than 0, a cost term will be added to the cost function to
    # minimize the ratio between the vertical ground reaction forces of the
    # front contact spheres over the vertical ground reaction forces of the
    # rear (heel) contact spheres. This might be used to discourage the
    # model to lean forward, which would reduce muscle effort.
    min_ratio_vGRF = False
    if 'min_ratio_vGRF' in settings:
       min_ratio_vGRF = settings['min_ratio_vGRF']
    if min_ratio_vGRF: 
        weights['vGRFRatioTerm'] = 1
        if 'vGRFRatioTerm' in settings['weights']:
            weights['vGRFRatioTerm'] = settings['weights']['vGRFRatioTerm']
          
    # If yCalcnToes is set to True, a constraint will enforce the 
    # vertical position of the origin of the calcaneus and toe segments
    # to be larger than yCalcnToesThresholds. This is used to prevent the
    # model to penetrate the ground, which might otherwise occur at the
    # begnning of the trial when no periodic constraints are enforced.
    yCalcnToes = False
    if 'yCalcnToes' in settings:
        yCalcnToes = settings['yCalcnToes']            
    if yCalcnToes :
        yCalcnToesThresholds = 0.015
        if 'yCalcnToesThresholds' in settings:
            yCalcnToesThresholds = settings['yCalcnToesThresholds']
            
    # Set filter_Qs_toTrack to True to filter the coordinate values to be
    # tracked with a cutoff frequency of cutoff_freq_Qs.
    filter_Qs_toTrack = True
    if 'filter_Qs_toTrack' in settings:
        filter_Qs_toTrack = settings['filter_Qs_toTrack']
    if filter_Qs_toTrack:
        cutoff_freq_Qs = 30 # default.
        if 'cutoff_freq_Qs' in settings:
            cutoff_freq_Qs = settings['cutoff_freq_Qs']
         
    # Set filter_Qds_toTrack to True to filter the coordinate speeds to be
    # tracked with a cutoff frequency of cutoff_freq_Qds.
    filter_Qds_toTrack = False
    if 'filter_Qds_toTrack' in settings:
        filter_Qds_toTrack =  settings['filter_Qds_toTrack']
    if filter_Qds_toTrack:
        cutoff_freq_Qds = 30 # default.
        if 'cutoff_freq_Qds' in settings:
            cutoff_freq_Qds = settings['cutoff_freq_Qds']
      
    # Set filter_Qdds_toTrack to True to filter the coordinate
    # accelerations to be tracked with a cutoff frequency of
    # cutoff_freq_Qdds.
    filter_Qdds_toTrack = False
    if 'filter_Qdds_toTrack' in settings:
        filter_Qdds_toTrack = settings['filter_Qdds_toTrack']
    if filter_Qdds_toTrack:
        cutoff_freq_Qdds = 30 # default.
        if 'cutoff_freq_Qdds' in settings:
            cutoff_freq_Qdds = settings['cutoff_freq_Qdds']
         
    # Set splineQds to True to compute the coordinate accelerations by
    # first splining the coordinate speeds and then taking the derivative.
    # The default approach is to spline the coordinate values and then 
    # take the second derivative. It might be useful to first filter the
    # coordinate speeds using filter_Qds_toTrack and then set splineQds
    # to True to obtain smoother coordinate accelerations.
    splineQds = False
    if 'splineQds' in settings:
        splineQds = settings['splineQds']
        
    # Set treadmill to True to simulate a treadmill motion. The treadmill
    # speed is given by treadmill_speed (positive is forward motion).
    treadmill = False
    if settings['treadmill_speed'] != 0:
        treadmill = True
    
    # Problem info.
    # Coordinates to track.
    coordinates_toTrack = settings['coordinates_toTrack']
    
    # Set offset_ty to True to include an optimization variable in the problem
    # that will offset the vertical pelvis position (pelvis_ty). This is used
    # when there is uncertainty in the size of the foot ground contact spheres,
    # which makes hard accurately tracking pelvis_ty without such offset.
    offset_ty = True
    if 'offset_ty' in settings:
        offset_ty = settings['offset_ty']
        
    # Set enableLimitTorques to True to include limit torques. This is used
    # to model the effect of ligaments at the extremes of the range of motion.
    # See passiveJointTorqueData_3D in muscleDataOpenSimAD for values.
    enableLimitTorques = False
    if 'enableLimitTorques' in settings:
        enableLimitTorques = settings['enableLimitTorques']    
        
    # Set periodicConstraints to True to include periodic constraints. These
    # constraints typically facilitate convergence, and we encourage users to
    # use them when appropriate. See code below for details about how to set
    # up periodic constraints for the problem states. See settingsOpenSimAD
    # for examples.
    periodicConstraints = False
    if 'periodicConstraints' in settings:
        periodicConstraints = settings['periodicConstraints']
        
    # Set trackQdds to track coordinate accelerations. We found this useful to
    # improve the convergence of the problem. The weight associated with the
    # cost term is given by weights['accelerationTrackingTerm'].
    trackQdds = True
    if 'trackQdds' in settings:
        trackQdds = settings['trackQdds']
    if trackQdds:
        weights['accelerationTrackingTerm'] = 1
        if 'accelerationTrackingTerm' in settings['weights']:
            weights['accelerationTrackingTerm'] = (
                settings['weights']['accelerationTrackingTerm'])
            
    # The variable powActivations determines the power at which muscle
    # activations are minimized. By default, we use a power of 2, but
    # researchers have used other powers in the past (eg, Ackermann and 
    # van den Bogert, 2010).
    powActivations = 2
    if 'powActivations' in settings:
        powActivations = settings['powActivations']
        
    # Set volumeScaling to True to scale individual muscle contributions to
    # the muscle effort term by their volume.
    volumeScaling = False
    if 'volumeScaling' in settings:
        volumeScaling = settings['volumeScaling']
        
    # Set coordinate_constraints to bound the joint coordinates (states) based
    # on the experimental data at each mesh point (time-based bounds). Eg,
    # coordinate_constraints['pelvis_tx'] = {"env_bound": 0.1} will bound the
    # design variable corresponding to the pelvis_tx value by 
    # "experimental Q value to be tracked +/- 0.1 m" at each mesh point;
    # coordinate_constraints['hip_flexion_l'] = {"env_bound": 0.1} will bound
    # the design variable corresponding to the hip_flexion_l value by 
    # "experimental Q value to be tracked +/- 0.1 rad" at each mesh point. This
    # is an experimental feature that might help convergence by reducing the
    # search space of the optimization problem. See settingsOpenSimAD for
    # examples.
    coordinate_constraints = {}
    if 'coordinate_constraints' in settings:
        coordinate_constraints = settings['coordinate_constraints']
        
    # Convergence tolerance of ipopt: 
    # See https://coin-or.github.io/Ipopt/OPTIONS.html for more details.
    # We recommend testing different tolerances to make sure the results are 
    # not impacted by too loose tolerances. In the examples, we did not find
    # differences between using a tolerance of 3 and 4, and therefore set it
    # to 3, since it converges faster. Nevertheless, we set it here to 4 by
    # default.
    ipopt_tolerance = 4
    if 'ipopt_tolerance' in settings:
        ipopt_tolerance = settings['ipopt_tolerance']

    # Set torque_driven_model to True to replace the muscle actuators with
    # ideal torque actuators. The default optimal forces of the ideal torque
    # actuators are defined in the function
    # get_coordinate_actuator_optimal_forces in muscleDataOpenSimAD. You can
    # also specify the optimal forces as part of the problem settings. For
    # example, to set the optimal force of the right hip flexion to 400, add
    # coordinate_optimal_forces['hip_flexion_r'] = 400 in the settings dict.
    # See example under running_torque_driven in settingsOpenSimAD.
    torque_driven_model = False
    if 'torque_driven_model' in settings:
        torque_driven_model = settings['torque_driven_model']
    if torque_driven_model:
        weights['coordinateExcitationTerm'] = 1
        if 'coordinateExcitationTerm' in settings['weights']:
            weights['coordinateExcitationTerm'] = (
                settings['weights']['coordinateExcitationTerm'])

    # Set useExpressionGraphFunction to True to use the expression graph
    # directly instead of compiling it to a function. This allows using the
    # CasADi type SX, which should be computationally more efficient. This is
    # an experimental feature.
    useExpressionGraphFunction = True
    if 'useExpressionGraphFunction' in settings:
        useExpressionGraphFunction = settings['useExpressionGraphFunction']

    # Set contact_side to 'left' or 'right' to only add contact spheres to the
    # specified side. This is useful when simulating motions with only one foot
    # in contact with the ground. The default is to add contact spheres to both
    # sides.
    contact_side = 'all'
    if 'contact_side' in settings:
        contact_side = settings['contact_side']

    print('Processing {} - Case {}'.format(trialName, case))

    # %% Paths and dirs.
    pathMain = os.getcwd()
    pathSubjectData = os.path.join(dataDir, subject)
    pathOSData = os.path.join(pathSubjectData, 'OpenSimData')
    pathModelFolder = os.path.join(pathOSData, 'Model')
    pathModelFile = os.path.join(pathModelFolder, model_full_name + ".osim")
    pathExternalFunctionFolder = os.path.join(pathModelFolder,
                                              'ExternalFunction')
    pathIKFolder = os.path.join(pathOSData, 'Kinematics')
    # If Force/EMG/IK/ID data is available, we load it for comparison.
    pathForceFolder = os.path.join(pathSubjectData, 'ForceData')
    pathEMGFolder = os.path.join(pathSubjectData, 'EMGData')    
    pathMocapFolder = os.path.join(pathSubjectData, "OpenSimDataMocap")
    pathMocapIKFolder = os.path.join(pathMocapFolder, 'InverseKinematics')
    pathMocapIDFolder = os.path.join(pathMocapFolder, 'InverseDynamics')
    # Path results and settings.
    pathResults = os.path.join(pathOSData, 'Dynamics', trialName)
    if 'repetition' in settings:
        pathResults = os.path.join(
            pathOSData, 'Dynamics', 
            trialName + '_rep' + str(settings['repetition']))     
    os.makedirs(pathResults, exist_ok=True)
    pathSettings = os.path.join(pathResults, 'Setup_{}.yaml'.format(case))
    # Dump settings in yaml file.
    with open(pathSettings, 'w') as file:
        yaml.dump(settings, file)
    
    # %% Muscles.
    # This section specifies the muscles and some of their parameters. This is
    # specific to the Rajagopal musculoskeletal model.
    #
    # WARNING: we do not use the muscle model defined in the .osim file. We
    # use our own muscle model based on De Groote et al. 2016:
    # https://pubmed.ncbi.nlm.nih.gov/27001399/. We only extract the muscle-
    # tendon parameters from the .osim file.
    muscles = [
        'addbrev_r', 'addlong_r', 'addmagDist_r', 'addmagIsch_r', 
        'addmagMid_r', 'addmagProx_r', 'bflh_r', 'bfsh_r', 'edl_r', 'ehl_r', 
        'fdl_r', 'fhl_r', 'gaslat_r', 'gasmed_r', 'glmax1_r', 'glmax2_r',
        'glmax3_r', 'glmed1_r', 'glmed2_r', 'glmed3_r', 'glmin1_r', 'glmin2_r',
        'glmin3_r', 'grac_r', 'iliacus_r', 'perbrev_r', 'perlong_r', 'piri_r', 
        'psoas_r', 'recfem_r', 'sart_r', 'semimem_r', 'semiten_r', 'soleus_r',
        'tfl_r', 'tibant_r', 'tibpost_r', 'vasint_r', 'vaslat_r', 'vasmed_r']
    rightSideMuscles = muscles        
    leftSideMuscles = [muscle[:-1] + 'l' for muscle in rightSideMuscles]
    bothSidesMuscles = leftSideMuscles + rightSideMuscles
    nMuscles = len(bothSidesMuscles)
    nSideMuscles = len(rightSideMuscles)
    
    # Extract muscle-tendon parameters (if not done already).
    from muscleDataOpenSimAD import getMTParameters
    loadMTParameters_l = True
    loadMTParameters_r = True
    if not os.path.exists(os.path.join(
            pathModelFolder, model_full_name + '_mtParameters_l.npy')):
        loadMTParameters_l = False
    if not os.path.exists(os.path.join(
            pathModelFolder, model_full_name + '_mtParameters_r.npy')):
        loadMTParameters_r = False
    righSideMtParameters = getMTParameters(pathModelFile, rightSideMuscles,
                                           loadMTParameters_r, pathModelFolder,
                                           model_full_name, side='r')
    leftSideMtParameters = getMTParameters(pathModelFile, leftSideMuscles,
                                           loadMTParameters_l, pathModelFolder,
                                           model_full_name, side='l')
    mtParameters = np.concatenate((leftSideMtParameters, 
                                   righSideMtParameters), axis=1)
    mtParameters[0,:] = mtParameters[0,:] * scaleIsometricMuscleForce
    
    # Tendon compliance.
    from muscleDataOpenSimAD import tendonCompliance
    sideTendonCompliance = tendonCompliance(nSideMuscles)
    tendonCompliance = np.concatenate((sideTendonCompliance, 
                                       sideTendonCompliance), axis=1)
    from muscleDataOpenSimAD import tendonShift
    sideTendonShift = tendonShift(nSideMuscles)
    tendonShift = np.concatenate((sideTendonShift, sideTendonShift), axis=1)
     
    # Specific tension.
    specificTension = 0.5*np.ones((1, nMuscles))
    
    # Hill-equilibrium. We use as muscle model the DeGrooteFregly2016 model
    # introduced in: https://pubmed.ncbi.nlm.nih.gov/27001399/.
    # In particular, we use the third formulation introduced in the paper,
    # with "normalized tendon force as a state and the scaled time derivative
    # of the normalized tendon force as a new control simplifying the
    # contraction dynamic equations".
    from functionCasADiOpenSimAD import hillEquilibrium    
    f_hillEquilibrium = hillEquilibrium(
        mtParameters, tendonCompliance, tendonShift, specificTension,
        ignorePassiveFiberForce=ignorePassiveFiberForce)
        
    # Time constants for activation dynamics.
    activationTimeConstant = 0.015
    deactivationTimeConstant = 0.06
    
    # Individual muscle weights. Option to weight the contribution of the
    # muscles to the muscle effort term differently. This is an experimental
    # feature.
    w_muscles = np.ones((nMuscles,1))
    if 'muscle_weights' in settings:
        for count, c_muscle in enumerate(bothSidesMuscles):
            if c_muscle in settings['muscle_weights']:
                w_muscles[count, 0] = (
                        settings['muscle_weights'][c_muscle]['weight'])
           
    # Muscle volume scaling.
    if volumeScaling:
        muscleVolume = np.multiply(mtParameters[0, :], mtParameters[1, :])
        s_muscleVolume = muscleVolume / np.sum(muscleVolume)
    else:
        s_muscleVolume = np.ones((nMuscles,))
    s_muscleVolume = np.reshape(s_muscleVolume, (nMuscles, 1))
    
    # Coordinate actuator optimal forces.
    from muscleDataOpenSimAD import get_coordinate_actuator_optimal_forces
    coordinate_optimal_forces = get_coordinate_actuator_optimal_forces()
    # Adjust based on settings.
    if 'coordinate_optimal_forces' in settings:
        for coord in settings['coordinate_optimal_forces']:
            coordinate_optimal_forces[coord] = (
                settings['coordinate_optimal_forces'][coord])
    
    # %% Coordinates.
    # This section specifies the coordinates of the model. This is
    # specific to the Rajagopal musculoskeletal model.
    from utilsOpenSimAD import getIndices
    joints = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx',
              'pelvis_ty', 'pelvis_tz', 'hip_flexion_l', 'hip_adduction_l',
              'hip_rotation_l', 'hip_flexion_r', 'hip_adduction_r',
              'hip_rotation_r', 'knee_angle_l', 'knee_angle_r',
              'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l',
              'subtalar_angle_r', 'mtp_angle_l', 'mtp_angle_r',
              'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    # Mtp coordinates.
    mtpJoints = ['mtp_angle_l', 'mtp_angle_r']
    nMtpJoints = len(mtpJoints)
    if not withMTP:
        for joint in mtpJoints:
            joints.remove(joint)
    # Lower-limb coordinates.
    lowerLimbJoints = copy.deepcopy(joints)
    # Arm coordinates.
    armJoints = ['arm_flex_l', 'arm_add_l', 'arm_rot_l',
                 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
                 'elbow_flex_l', 'elbow_flex_r', 'pro_sup_l', 'pro_sup_r']
    nArmJoints = len(armJoints)
    if withArms:
        for joint in armJoints:
            joints.append(joint)
    # Total count.
    nJoints = len(joints)
    
    # Translational coordinates.
    translationalJoints = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    
    # Rotational coordinates.
    rotationalJoints = copy.deepcopy(joints)
    for joint in translationalJoints:
        rotationalJoints.remove(joint)
    idxRotationalJoints = getIndices(joints, rotationalJoints)
    
    # Ground pelvis coordinates.
    groundPelvisJoints = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                          'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    groundPelvisJointsForces = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    
    # Lumbar coordinates (for torque actuators).    
    lumbarJoints = ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    nLumbarJoints = len(lumbarJoints)
    
    # Coordinates with passive torques.
    passiveTorqueJoints = [
        'hip_flexion_r', 'hip_flexion_l', 'hip_adduction_r', 'hip_adduction_l',
        'hip_rotation_r', 'hip_rotation_l', 'knee_angle_r', 'knee_angle_l', 
        'ankle_angle_r', 'ankle_angle_l', 'subtalar_angle_r', 
        'subtalar_angle_l', 'mtp_angle_r', 'mtp_angle_l',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    if not withMTP:
        for joint in mtpJoints:
            passiveTorqueJoints.remove(joint)
    nPassiveTorqueJoints = len(passiveTorqueJoints)

    # Muscle-driven coordinates.
    muscleDrivenJoints = [
        'hip_flexion_l', 'hip_flexion_r', 'hip_adduction_l',
        'hip_adduction_r', 'hip_rotation_l', 'hip_rotation_r', 'knee_angle_l',
        'knee_angle_r', 'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l',
        'subtalar_angle_r', 'lumbar_extension', 'lumbar_bending',
        'lumbar_rotation']
    for joint in lumbarJoints:
        muscleDrivenJoints.remove(joint)
    nMuscleDrivenJoints = len(muscleDrivenJoints)
        
    # Specify which states should have periodic constraints.
    # See settingsOpenSimAD for example.
    if periodicConstraints:        
        if 'coordinateValues' in periodicConstraints:
            if 'lowerLimbJoints' in periodicConstraints['coordinateValues']:
                idxPeriodicQs = getIndices(joints, lowerLimbJoints)
            else:
                idxPeriodicQs = []
                for joint in periodicConstraints['coordinateValues']:
                    idxPeriodicQs.append(joints.index(joint))
        if 'coordinateSpeeds' in periodicConstraints:
            if 'lowerLimbJoints' in periodicConstraints['coordinateSpeeds']:
                idxPeriodicQds = getIndices(joints, lowerLimbJoints)
            else:
                idxPeriodicQds = []
                for joint in periodicConstraints['coordinateSpeeds']:
                    idxPeriodicQds.append(joints.index(joint))

        if 'muscleActivationsForces' in periodicConstraints:
            if 'all' in periodicConstraints['muscleActivationsForces']:
                idxPeriodicMuscles = getIndices(bothSidesMuscles, 
                                                bothSidesMuscles)
            else:
                idxPeriodicMuscles = []
                for c_m in periodicConstraints['muscleActivationsForces']:
                    idxPeriodicMuscles.append(bothSidesMuscles.index(c_m))
        if 'lumbarJointActivations' in periodicConstraints:
            if 'all' in periodicConstraints['lumbarJointActivations']:
                idxPeriodicLumbar = getIndices(lumbarJoints, 
                                                lumbarJoints)
            else:
                idxPeriodicLumbar = []
                for c_m in periodicConstraints['lumbarJointActivations']:
                    idxPeriodicLumbar.append(lumbarJoints.index(c_m))

        if ('lowerLimbJointActivations' in periodicConstraints 
            and torque_driven_model):
            if 'all' in periodicConstraints['lowerLimbJointActivations']:
                    idxPeriodicMuscles = getIndices(muscleDrivenJoints, 
                                                    muscleDrivenJoints)
            else:
                idxPeriodicMuscles = []
                for c_m in periodicConstraints['lowerLimbJointActivations']:
                    idxPeriodicMuscles.append(muscleDrivenJoints.index(c_m))
    
        
    # %% Coordinate actuator activation dynamics.
    if withArms or withLumbarCoordinateActuators:
        from functionCasADiOpenSimAD import coordinateActuatorDynamics
        if withArms:
            f_armDynamics = coordinateActuatorDynamics(nArmJoints)
        if withLumbarCoordinateActuators:
            f_lumbarDynamics = coordinateActuatorDynamics(nLumbarJoints)
        if torque_driven_model:
            f_coordinateDynamics = coordinateActuatorDynamics(nMuscleDrivenJoints)
    
    # %% Passive/limit torques.
    from functionCasADiOpenSimAD import limitPassiveTorque, linarPassiveTorque
    from muscleDataOpenSimAD import passiveJointTorqueData
    damping = 0.1
    f_passiveTorque = {}
    for joint in passiveTorqueJoints:
        f_passiveTorque[joint] = limitPassiveTorque(
            passiveJointTorqueData(joint)[0],
            passiveJointTorqueData(joint)[1], damping)    
    if withMTP:
        stiffnessMtp = 25
        dampingMtp = 2
        f_linearPassiveMtpTorque = linarPassiveTorque(stiffnessMtp, dampingMtp)        
    if withArms:
        stiffnessArm = 0
        dampingArm = 0.1
        f_linearPassiveArmTorque = linarPassiveTorque(stiffnessArm, dampingArm)
        
    # %% Kinematic data to track.
    from utilsOpenSimAD import getIK, filterDataFrame
    from utilsOpenSimAD import interpolateDataFrame, selectDataFrame
    pathIK = os.path.join(pathIKFolder, trialName + '.mot')
    Qs_fromIK = getIK(pathIK, joints)
    # Filtering.
    if filter_Qs_toTrack:
        Qs_fromIK_filter = filterDataFrame(
            Qs_fromIK, cutoff_frequency=cutoff_freq_Qs)
    else:
        Qs_fromIK_filter = Qs_fromIK           
    # Interpolation.
    Qs_toTrack = interpolateDataFrame(
        Qs_fromIK_filter, timeIntervals[0], timeIntervals[1], N)
    # We do not want to down-sample before differentiating the splines.
    Qs_fromIK_sel = selectDataFrame(
        Qs_fromIK_filter, timeIntervals[0], timeIntervals[1])
    nEl_toTrack = len(coordinates_toTrack)
    # Individual coordinate weigths. Option to weight the contribution of the
    # coordinates to the cost terms differently. This applies for coordinate
    # values, speeds, and accelerations tracking.
    w_dataToTrack = np.ones((nEl_toTrack,1))
    coordinates_toTrack_l = []
    for count, coord in enumerate(coordinates_toTrack):
        coordinates_toTrack_l.append(coord)
        if 'weight' in coordinates_toTrack[coord]:
            w_dataToTrack[count, 0] = coordinates_toTrack[coord]['weight']      
    idx_coordinates_toTrack = getIndices(joints, coordinates_toTrack_l)
    
    from utilsOpenSimAD import scaleDataFrame, selectFromDataFrame     
    dataToTrack_Qs_nsc = selectFromDataFrame(
        Qs_toTrack, coordinates_toTrack_l).to_numpy()[:,1::].T
    
    # If Force/EMG/IK/ID data is available, we load it for comparison.
    # Force (GRF and GRM)
    from utilsOpenSimAD import getGRFAll, getEMG, getID
    pathGRFFile = os.path.join(pathForceFolder, trialName + '.mot')
    experimental_force_data_available = False
    if os.path.exists(pathGRFFile):
        experimental_force_data_available = True
        experimental_force = getGRFAll(pathGRFFile, timeIntervals, N)
        experimental_grf = selectFromDataFrame(
            experimental_force['df_interp']['forces']['all'], 
            experimental_force['headers']['forces']['all']).to_numpy()[:,1::].T        
        experimental_grm = selectFromDataFrame(
            experimental_force['df_interp']['torques_G']['all'], 
            experimental_force['headers']['torques']['all']).to_numpy()[:,1::].T
    # EMG
    pathEMGFile = os.path.join(pathEMGFolder, trialName + '.sto')
    experimental_emg_data_available = False
    if os.path.exists(pathEMGFile):
        experimental_emg_data_available = True
        experimental_emg_t = getEMG(pathEMGFile, bothSidesMuscles)
        experimental_emg = interpolateDataFrame(experimental_emg_t, 
            timeIntervals[0], timeIntervals[1], N).to_numpy()[:,1::].T
    # IK
    pathIKFileMocap = os.path.join(pathMocapIKFolder, trialName + '.mot')
    mocap_ik_data_available = False
    if os.path.exists(pathIKFileMocap):
        mocap_ik_data_available = True
        Qs_mocap = getIK(pathIKFileMocap, joints)
        if filter_Qs_toTrack:
            Qs_mocap_filter = filterDataFrame(
                Qs_mocap, cutoff_frequency=cutoff_freq_Qs)
        else:
            Qs_mocap_filter = Qs_mocap
        Qs_mocap_sel = selectDataFrame(
                Qs_mocap_filter, timeIntervals[0], timeIntervals[1])
        Qs_mocap_ref = interpolateDataFrame(
            Qs_mocap_filter, timeIntervals[0], timeIntervals[1],
            N).to_numpy()[:,1::].T
    # ID
    pathIDFile = os.path.join(pathMocapIDFolder, trialName + '.sto')
    mocap_id_data_available = False
    if os.path.exists(pathIDFile):
        mocap_id_data_available = True
        mocap_id_t = getID(pathIDFile, joints)
        mocap_id = interpolateDataFrame(
            mocap_id_t, timeIntervals[0], timeIntervals[1], N).to_numpy()[:,1::].T
            
    # %% Polynomial approximations.
    # Muscle-tendon lengths, velocities, and moment arms are estimated based
    # on polynomial approximations of joint positions and velocities. The
    # polynomial coefficients are fitted based on data from OpenSim and saved
    # for the current model.
    
    # Paths
    pathGenericTemplates = os.path.join(baseDir, "OpenSimPipeline")
    pathDummyMotion = os.path.join(pathGenericTemplates, "MuscleAnalysis", 
                                   'DummyMotion.mot')
    
    # These are the ranges of motion used to fit the polynomial coefficients.
    # We do not want the experimental data to be out of these ranges. If they
    # are, we make them larger and fit polynomial coefficients specific to the
    # trial being processed. These are also the bounds used in the optimal
    # control problem.
    polynomial_bounds = {
        'hip_flexion_l': {'max': 120, 'min': -30},
        'hip_flexion_r': {'max': 120, 'min': -30},
        'hip_adduction_l': {'max': 20, 'min': -50},
        'hip_adduction_r': {'max': 20, 'min': -50},
        'hip_rotation_l': {'max': 35, 'min': -40},
        'hip_rotation_r': {'max': 35, 'min': -40},
        'knee_angle_l': {'max': 138, 'min': 0},
        'knee_angle_r': {'max': 138, 'min': 0},
        'knee_adduction_l': {'max': 20, 'min': -30},
        'knee_adduction_r': {'max': 20, 'min': -30},
        'ankle_angle_l': {'max': 50, 'min': -50},
        'ankle_angle_r': {'max': 50, 'min': -50},
        'subtalar_angle_l': {'max': 35, 'min': -35},
        'subtalar_angle_r': {'max': 35, 'min': -35},
        'mtp_angle_l': {'max': 5, 'min': -45},
        'mtp_angle_r': {'max': 5, 'min': -45}}
    model_bounds = {
        'hip_flexion_l': {'max': 120, 'min': -30},
        'hip_flexion_r': {'max': 120, 'min': -30},
        'hip_adduction_l': {'max': 30, 'min': -50},
        'hip_adduction_r': {'max': 30, 'min': -50},
        'hip_rotation_l': {'max': 40, 'min': -40},
        'hip_rotation_r': {'max': 40, 'min': -40},
        'knee_angle_l': {'max': 140, 'min': 0},
        'knee_angle_r': {'max': 140, 'min': 0},
        'knee_adduction_l': {'max': 20, 'min': -30},
        'knee_adduction_r': {'max': 20, 'min': -30},
        'ankle_angle_l': {'max': 50, 'min': -50},
        'ankle_angle_r': {'max': 50, 'min': -50},
        'subtalar_angle_l': {'max': 35, 'min': -35},
        'subtalar_angle_r': {'max': 35, 'min': -35},
        'mtp_angle_l': {'max': 30, 'min': -45},
        'mtp_angle_r': {'max': 30, 'min': -45}}
    # Check if the Qs (coordinate values) to track are within the bounds
    # used to define the polynomials. If not, adjust the polynomial bounds.
    from utilsOpenSimAD import checkQsWithinPolynomialBounds
    updated_bounds = checkQsWithinPolynomialBounds(
        dataToTrack_Qs_nsc, polynomial_bounds, model_bounds, coordinates_toTrack_l)
    type_bounds_polynomials = 'default'
    if len(updated_bounds) > 0:
        # Modify the values of polynomial_bounds based on the values in
        # updated_bounds.  Also, create a dummy motion file specific to the
        # trial being processed.
        from utilsOpenSimAD import adjustBoundsAndDummyMotion
        polynomial_bounds, pathDummyMotion = adjustBoundsAndDummyMotion(
            polynomial_bounds, updated_bounds, pathDummyMotion,
            pathModelFolder, trialName, overwriteDummyMotion=False)
        type_bounds_polynomials = trialName
    
    from functionCasADiOpenSimAD import polynomialApproximation
    leftPolynomialJoints = [
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
        'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'] 
    rightPolynomialJoints = [
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
        'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r']
    if not withMTP:
        leftPolynomialJoints.remove('mtp_angle_l')
        rightPolynomialJoints.remove('mtp_angle_r')    
    
    if not torque_driven_model:
        # Load polynomials if computed already, compute otherwise.    
        loadPolynomialData = True
        if (not os.path.exists(os.path.join(
                pathModelFolder, model_full_name + '_polynomial_r_{}.npy'.format(type_bounds_polynomials)))
                or not os.path.exists(os.path.join(
                pathModelFolder, model_full_name + '_polynomial_l_{}.npy'.format(type_bounds_polynomials)))):
            loadPolynomialData = False        
        from muscleDataOpenSimAD import getPolynomialData
        polynomialData = {}
        polynomialData['r'] = getPolynomialData(
            loadPolynomialData, pathModelFolder, model_full_name, pathDummyMotion, 
            rightPolynomialJoints, rightSideMuscles, 
            type_bounds_polynomials=type_bounds_polynomials, side='r')
        polynomialData['l'] = getPolynomialData(
            loadPolynomialData, pathModelFolder, model_full_name, pathDummyMotion, 
            leftPolynomialJoints, leftSideMuscles, 
            type_bounds_polynomials=type_bounds_polynomials, side='l')     
        if loadPolynomialData:
            polynomialData['r'] = polynomialData['r'].item()
            polynomialData['l'] = polynomialData['l'].item()
        # Coefficients should not be larger than 1.
        sides = ['r', 'l']
        polynomialCheck = True
        for side in sides:
            for c_pol in polynomialData[side]:
                if np.max(polynomialData[side][c_pol]['coefficients']) > 1:
                    polynomialCheck = False
        if not polynomialCheck:
            # TODO: I don't think that this is a problem.
            print('Some polynomial coefficients are larger than 1.')
                
        # The function f_polynomial takes as inputs joint positions and velocities
        # from one side, and returns muscle-tendon lengths, velocities, and moment
        # arms for the muscle of that side.
        nPolynomials = len(leftPolynomialJoints)
        f_polynomial = {}
        f_polynomial['r'] = polynomialApproximation(
            rightSideMuscles, polynomialData['r'], nPolynomials)
        f_polynomial['l'] = polynomialApproximation(
            leftSideMuscles, polynomialData['l'], nPolynomials)
        
        # Helper indices.
        leftPolynomialJointIndices = getIndices(joints, leftPolynomialJoints)
        rightPolynomialJointIndices = getIndices(joints, rightPolynomialJoints)    
        leftPolynomialMuscleIndices = (
            list(range(nSideMuscles)) + 
            list(range(nSideMuscles, nSideMuscles)))
        rightPolynomialMuscleIndices = list(range(nSideMuscles))
        from utilsOpenSimAD import getMomentArmIndices
        momentArmIndices = getMomentArmIndices(
            rightSideMuscles, leftPolynomialJoints,rightPolynomialJoints, 
            polynomialData['r'])    
    
        # Plot polynomial approximations (when possible) for sanity check.
        plotPolynomials = False
        if plotPolynomials:
            from polynomialsOpenSimAD import testPolynomials
            path_data4PolynomialFitting = os.path.join(
                pathModelFolder, 
                'data4PolynomialFitting_{}_{}.npy'.format(
                    model_full_name, type_bounds_polynomials))
            data4PolynomialFitting = np.load(path_data4PolynomialFitting, 
                                             allow_pickle=True).item()            
            testPolynomials(
                data4PolynomialFitting, rightPolynomialJoints, rightSideMuscles,
                f_polynomial['r'], polynomialData['r'], momentArmIndices)
            testPolynomials(
                data4PolynomialFitting, leftPolynomialJoints, leftSideMuscles,
                f_polynomial['l'], polynomialData['l'], momentArmIndices)
    
    # %% External functions.
    # The external function builds the OpenSim model and run inverse dynamics.
    # The function takes as inputs joint positions, velocities, and 
    # accelerations, which are states and controls of the optimal control 
    # problem. It returns joint torques as well as some outputs of interest, 
    # eg segment origins, that you may want to use for the problem formulation.
    # The external function is written in C++ and compiled as an executable,
    # which when called with numerical values returns the underlying expression
    # graph as a function foo. We used to generate c code from the expression
    # graph using CasADi and then compile the c code as a library to be called
    # as an external function. This is not necessary anymore. The expression
    # graph is saved as a function that can be called directly with CasADi. This
    # allows using SX only, whereas actual external functions require MX. We
    # still support the older approach (useExpressionGraphFunction=False).

    F_name = 'F'
    dim = 3*nJoints
    if treadmill:
        F_name += '_treadmill'
        dim += 1
    if contact_side != 'all':
        F_name += '_{}'.format(contact_side)
    if useExpressionGraphFunction:
        from utilsOpenSimAD import getF_expressingGraph
        # Import function for expression graph.    
        sys.path.append(pathExternalFunctionFolder)
        os.chdir(pathExternalFunctionFolder)    
        F = getF_expressingGraph(dim, F_name)
        sys.path.remove(pathExternalFunctionFolder)
        os.chdir(pathMain)
    else: # This will be deprecated
        if platform.system() == 'Windows':
            ext_F = '.dll'
        elif platform.system() == 'Darwin':
            ext_F = '.dylib'
        elif platform.system() == 'Linux':
            ext_F = '.so'
        else:
            raise ValueError("Platform not supported.")
        F = ca.external('F', 
            os.path.join(pathExternalFunctionFolder, F_name + ext_F))
    # F_map contains information about input/output of the function.
    F_map = np.load(
        os.path.join(pathExternalFunctionFolder, 
                     F_name + '_map.npy'), allow_pickle=True).item()
    
    # Indices outputs external function.
    if 'nContactSpheres' not in F_map['GRFs']:
        # We updated the code to make it more generic and allow for different
        # contact configurations. Old versions of the external functions will
        # not work anymore. Results will not change.
        raise ValueError("""We recently updated our code, please delete the folder 
        ExternalFunction under Data/<session_ID>/OpenSimData/Model/ 
        and rerun the example_kinetics.py.""")
        
    nContactSpheres = F_map['GRFs']['nContactSpheres']
    contactSpheres = {}
    contactSpheres['right'] = F_map['GRFs']['rightContactSpheres']
    contactSpheres['left'] = F_map['GRFs']['leftContactSpheres']    
    contactSpheres['all'] = contactSpheres['right'] + contactSpheres['left']
    contactSpheres['bodies'] = {}
    contactSpheres['bodies']['right']  = F_map['GRFs']['rightContactSphereBodies']
    contactSpheres['bodies']['left'] = F_map['GRFs']['leftContactSphereBodies']
    contactSides = []
    if contactSpheres['right']:
        contactSides.append('right')
    if contactSpheres['left']:
        contactSides.append('left')        
    if heel_vGRF_threshold > 0:
        # Indices vertical ground reaction forces heel contact spheres.
        idx_vGRF_heel = []
        for side in contactSides:
            idx_vGRF_heel.append(F_map['GRFs'][contactSpheres[side][0]][1])
    if min_ratio_vGRF:            
        # Indices vertical ground reaction forces rear contact spheres.
        idx_vGRF_rear, idx_vGRF_front = {}, {}
        # Warning: hard coded
        idx_rear_spheres = [0, 3]
        idx_front_spheres = [1, 2, 4, 5]
        for side in contactSides:
            idx_vGRF_rear[side] = [F_map['GRFs'][contactSpheres[side][i]][1] for i in idx_rear_spheres]
            idx_vGRF_front[side] = [F_map['GRFs'][contactSpheres[side][i]][1] for i in idx_front_spheres]        
    if yCalcnToes:
        # Indices vertical position origins calc and toes segments.
        idx_yCalcnToes = [F_map['body_origins']['calcn_l'][1],
                          F_map['body_origins']['calcn_r'][1],
                          F_map['body_origins']['toes_l'][1],
                          F_map['body_origins']['toes_r'][1]]

    # Lists to map order of coordinates defined here and in external function.
    idxGroundPelvisJointsinF = [F_map['residuals'][joint] 
                                for joint in groundPelvisJoints]    
    idxJoints4F = [joints.index(joint) 
                   for joint in list(F_map['residuals'].keys())]    
   
    # %% Helper CasADi functions
    from functionCasADiOpenSimAD import normSumSqr
    from functionCasADiOpenSimAD import normSumWeightedPow
    from functionCasADiOpenSimAD import diffTorques
    from functionCasADiOpenSimAD import normSumWeightedSqrDiff
    f_NMusclesSum2 = normSumSqr(nMuscles)
    f_NMusclesSumWeightedPow = normSumWeightedPow(nMuscles, powActivations)
    f_nJointsSum2 = normSumSqr(nJoints)
    f_NQsToTrackWSum2 = normSumWeightedSqrDiff(nEl_toTrack)
    if withArms:
        f_nArmJointsSum2 = normSumSqr(nArmJoints)
    if withLumbarCoordinateActuators:
        f_nLumbarJointsSum2 = normSumSqr(nLumbarJoints)
    if torque_driven_model:
        f_nCoordinatesSum2 = normSumSqr(nMuscleDrivenJoints)
    f_diffTorques = diffTorques()  
    
    # %% OPTIMAL CONTROL PROBLEM FORMULATION
    # We use an orthogonal third-order radau collocation scheme.
    d = 3 # interpolating polynomial.
    tau = ca.collocation_points(d,'radau')
    [C,D] = ca.collocation_interpolators(tau) # collocation matrices.
    # Missing B matrix, adding manually. See CasADi example for code.
    # https://web.casadi.org/
    if d == 3:  
        B = [0, 0.376403062700467, 0.512485826188421, 0.111111111111111]
    elif d == 2:
        B = [0, 0.75, 0.25]
        
    # %% Bounds of the optimal control problem.
    from boundsOpenSimAD import bounds_tracking
    # Pre-allocations.
    uw, lw, scaling = {}, {}, {}
    bounds = bounds_tracking(Qs_toTrack, joints, rightSideMuscles)
    # States.
    if torque_driven_model:
        # Coordinate activations.
        uw['CoordA'], lw['CoordA'], scaling['CoordA'] = bounds.getBoundsCoordinateDynamics(muscleDrivenJoints, coordinate_optimal_forces)
        uw['CoordAk'] = ca.vec(uw['CoordA'].to_numpy().T * np.ones((1, N+1))).full()
        lw['CoordAk'] = ca.vec(lw['CoordA'].to_numpy().T * np.ones((1, N+1))).full()
        uw['CoordAj'] = ca.vec(uw['CoordA'].to_numpy().T * np.ones((1, d*N))).full()
        lw['CoordAj'] = ca.vec(lw['CoordA'].to_numpy().T * np.ones((1, d*N))).full()
    else:
        # Muscle activations.
        uw['A'], lw['A'], scaling['A'] = bounds.getBoundsActivation(lb_activation=lb_activation)
        uw['Ak'] = ca.vec(uw['A'].to_numpy().T * np.ones((1, N+1))).full()
        lw['Ak'] = ca.vec(lw['A'].to_numpy().T * np.ones((1, N+1))).full()
        uw['Aj'] = ca.vec(uw['A'].to_numpy().T * np.ones((1, d*N))).full()
        lw['Aj'] = ca.vec(lw['A'].to_numpy().T * np.ones((1, d*N))).full()
        # Muscle forces.
        uw['F'], lw['F'], scaling['F'] = bounds.getBoundsForce()
        uw['Fk'] = ca.vec(uw['F'].to_numpy().T * np.ones((1, N+1))).full()
        lw['Fk'] = ca.vec(lw['F'].to_numpy().T * np.ones((1, N+1))).full()
        uw['Fj'] = ca.vec(uw['F'].to_numpy().T * np.ones((1, d*N))).full()
        lw['Fj'] = ca.vec(lw['F'].to_numpy().T * np.ones((1, d*N))).full()
    # Joint positions.
    uw['Qs'], lw['Qs'], scaling['Qs'] =  bounds.getBoundsPosition(polynomial_bounds)
    uw['Qsk'] = ca.vec(uw['Qs'].to_numpy().T * np.ones((1, N+1))).full()
    lw['Qsk'] = ca.vec(lw['Qs'].to_numpy().T * np.ones((1, N+1))).full()
    uw['Qsj'] = ca.vec(uw['Qs'].to_numpy().T * np.ones((1, d*N))).full()
    lw['Qsj'] = ca.vec(lw['Qs'].to_numpy().T * np.ones((1, d*N))).full()
    # Joint velocities.
    uw['Qds'], lw['Qds'], scaling['Qds'] = bounds.getBoundsVelocity()
    uw['Qdsk'] = ca.vec(uw['Qds'].to_numpy().T*np.ones((1, N+1))).full()
    lw['Qdsk'] = ca.vec(lw['Qds'].to_numpy().T*np.ones((1, N+1))).full()
    uw['Qdsj'] = ca.vec(uw['Qds'].to_numpy().T*np.ones((1, d*N))).full()
    lw['Qdsj'] = ca.vec(lw['Qds'].to_numpy().T*np.ones((1, d*N))).full()    
    if withArms:
        # Arm activations.
        uw['ArmA'], lw['ArmA'], scaling['ArmA'] = bounds.getBoundsCoordinateDynamics(armJoints, coordinate_optimal_forces)
        uw['ArmAk'] = ca.vec(uw['ArmA'].to_numpy().T * np.ones((1, N+1))).full()
        lw['ArmAk'] = ca.vec(lw['ArmA'].to_numpy().T * np.ones((1, N+1))).full()
        uw['ArmAj'] = ca.vec(uw['ArmA'].to_numpy().T * np.ones((1, d*N))).full()
        lw['ArmAj'] = ca.vec(lw['ArmA'].to_numpy().T * np.ones((1, d*N))).full()
    if withLumbarCoordinateActuators:
        # Lumbar activations.
        uw['LumbarA'], lw['LumbarA'], scaling['LumbarA'] = bounds.getBoundsCoordinateDynamics(lumbarJoints, coordinate_optimal_forces)
        uw['LumbarAk'] = ca.vec(uw['LumbarA'].to_numpy().T * np.ones((1, N+1))).full()
        lw['LumbarAk'] = ca.vec(lw['LumbarA'].to_numpy().T * np.ones((1, N+1))).full()
        uw['LumbarAj'] = ca.vec(uw['LumbarA'].to_numpy().T * np.ones((1, d*N))).full()
        lw['LumbarAj'] = ca.vec(lw['LumbarA'].to_numpy().T * np.ones((1, d*N))).full()    
    # Controls.
    if torque_driven_model:
        # Coordinate excitations.
        uw['CoordE'], lw['CoordE'], scaling['CoordE'] = bounds.getBoundsCoordinateDynamics(muscleDrivenJoints, coordinate_optimal_forces)
        uw['CoordEk'] = ca.vec(uw['CoordE'].to_numpy().T * np.ones((1, N))).full()
        lw['CoordEk'] = ca.vec(lw['CoordE'].to_numpy().T * np.ones((1, N))).full()
    else:
        # Muscle activation derivatives.
        uw['ADt'], lw['ADt'], scaling['ADt'] = bounds.getBoundsActivationDerivative(
            activationTimeConstant=activationTimeConstant,
            deactivationTimeConstant=deactivationTimeConstant)
        uw['ADtk'] = ca.vec(uw['ADt'].to_numpy().T * np.ones((1, N))).full()
        lw['ADtk'] = ca.vec(lw['ADt'].to_numpy().T * np.ones((1, N))).full()
        # Muscle force derivatives.      
        uw['FDt'], lw['FDt'], scaling['FDt'] = bounds.getBoundsForceDerivative()
        uw['FDtk'] = ca.vec(uw['FDt'].to_numpy().T * np.ones((1, N))).full()
        lw['FDtk'] = ca.vec(lw['FDt'].to_numpy().T * np.ones((1, N))).full()
    if withArms:
        # Arm excitations.
        uw['ArmE'], lw['ArmE'], scaling['ArmE'] = bounds.getBoundsCoordinateDynamics(armJoints, coordinate_optimal_forces)
        uw['ArmEk'] = ca.vec(uw['ArmE'].to_numpy().T * np.ones((1, N))).full()
        lw['ArmEk'] = ca.vec(lw['ArmE'].to_numpy().T * np.ones((1, N))).full()
    if withLumbarCoordinateActuators:
        # Lumbar excitations.
        uw['LumbarE'], lw['LumbarE'], scaling['LumbarE'] = bounds.getBoundsCoordinateDynamics(lumbarJoints, coordinate_optimal_forces)
        uw['LumbarEk'] = ca.vec(uw['LumbarE'].to_numpy().T * np.ones((1, N))).full()
        lw['LumbarEk'] = ca.vec(lw['LumbarE'].to_numpy().T * np.ones((1, N))).full()    
    # Joint velocity derivatives (accelerations).
    uw['Qdds'], lw['Qdds'], scaling['Qdds'] = bounds.getBoundsAcceleration()
    uw['Qddsk'] = ca.vec(uw['Qdds'].to_numpy().T * np.ones((1, N))).full()
    lw['Qddsk'] = ca.vec(lw['Qdds'].to_numpy().T * np.ones((1, N))).full()
    # Reserve actuators.
    if withReserveActuators:
        uw['rAct'], lw['rAct'], scaling['rAct'] = {}, {}, {}
        uw['rActk'], lw['rActk'], = {}, {}
        for c_j in reserveActuatorCoordinates:
            uw['rAct'][c_j], lw['rAct'][c_j], scaling['rAct'][c_j] = (
                bounds.getBoundsReserveActuators(c_j, reserveActuatorCoordinates[c_j]))
            uw['rActk'][c_j] = ca.vec(uw['rAct'][c_j].to_numpy().T * np.ones((1, N))).full()
            lw['rActk'][c_j] = ca.vec(lw['rAct'][c_j].to_numpy().T * np.ones((1, N))).full()                
    # Static parameters.
    if offset_ty:
        scaling['Offset'] = 1.         
        uw['Offset'], lw['Offset'] = bounds.getBoundsOffset(scaling['Offset'])
        uw['Offsetk'] = uw['Offset'].to_numpy()
        lw['Offsetk'] = lw['Offset'].to_numpy()
    
    # %% Initial guess of the optimal control problem.
    from initialGuessOpenSimAD import dataDrivenGuess_tracking
    # Pre-allocations.
    w0 = {}  
    guess = dataDrivenGuess_tracking(
        Qs_toTrack, N, d, joints, bothSidesMuscles)
    # States.
    if torque_driven_model:
        # Coordinate activations.
        w0['CoordA'] = guess.getGuessTorqueActuatorActivation(muscleDrivenJoints)   
        w0['CoordAj'] = guess.getGuessTorqueActuatorActivationCol(muscleDrivenJoints)
    else:
        # Muscle activations.
        w0['A'] = guess.getGuessActivation(scaling['A'])
        w0['Aj'] = guess.getGuessActivationCol()
        # Muscle forces.
        w0['F'] = guess.getGuessForce(scaling['F'])
        w0['Fj'] = guess.getGuessForceCol()
    # Joint positions.
    w0['Qs'] = guess.getGuessPosition(scaling['Qs'])
    w0['Qsj'] = guess.getGuessPositionCol()
    # Joint velocities.
    w0['Qds'] = guess.getGuessVelocity(scaling['Qds'])
    w0['Qdsj'] = guess.getGuessVelocityCol()    
    if withArms:
        # Arm activations.
        w0['ArmA'] = guess.getGuessTorqueActuatorActivation(armJoints)   
        w0['ArmAj'] = guess.getGuessTorqueActuatorActivationCol(armJoints)
    if withLumbarCoordinateActuators:
        # Lumbar activations.
        w0['LumbarA'] = guess.getGuessTorqueActuatorActivation(lumbarJoints)   
        w0['LumbarAj'] = guess.getGuessTorqueActuatorActivationCol(lumbarJoints)    
    # Controls
    if torque_driven_model:
        # Coordinate excitations.
        w0['CoordE'] = guess.getGuessTorqueActuatorExcitation(muscleDrivenJoints)
    else:
        # Muscle activation derivatives.
        w0['ADt'] = guess.getGuessActivationDerivative(scaling['ADt'])
        # Muscle force derivatives.
        w0['FDt'] = guess.getGuessForceDerivative(scaling['FDt'])
    if withArms:
        # Arm excitations.
        w0['ArmE'] = guess.getGuessTorqueActuatorExcitation(armJoints)
    if withLumbarCoordinateActuators:
        # Lumbar excitations.
        w0['LumbarE'] = guess.getGuessTorqueActuatorExcitation(lumbarJoints)    
    # Joint velocity derivatives (accelerations).
    w0['Qdds'] = guess.getGuessAcceleration(scaling['Qdds'])
    # Reserve actuators.
    if withReserveActuators:
        w0['rAct'] = {}
        for c_j in reserveActuatorCoordinates:
            w0['rAct'][c_j] = guess.getGuessReserveActuators(c_j)
    # Static parameters.
    if offset_ty:
        w0['Offset'] = guess.getGuessOffset(scaling['Offset'])
            
    # %% Process tracking data.
    # Initialize dataframes.  
    Qs_spline = pd.DataFrame(columns=Qs_fromIK_sel.columns)
    Qds_spline = pd.DataFrame(columns=Qs_fromIK_sel.columns)
    Qdds_spline = pd.DataFrame(columns=Qs_fromIK_sel.columns)
    # Set time.
    Qs_spline['time'] = Qs_fromIK_sel['time']
    Qds_spline['time'] = Qs_fromIK_sel['time']
    Qdds_spline['time'] = Qs_fromIK_sel['time']
    # Compute values.
    for joint in joints:
        spline = interpolate.InterpolatedUnivariateSpline(
            Qs_fromIK_sel['time'], Qs_fromIK_sel[joint], k=3)
        Qs_spline[joint] = spline(Qs_fromIK_sel['time'])
        splineD1 = spline.derivative(n=1)
        Qds_spline[joint] = splineD1(Qs_fromIK_sel['time'])    
        splineD2 = spline.derivative(n=2)
        Qdds_spline[joint] = splineD2(Qs_fromIK_sel['time'])        
    # Filter.
    if filter_Qds_toTrack:
        Qds_spline_filter = filterDataFrame(
            Qds_spline, cutoff_frequency=cutoff_freq_Qds)
    else:
        Qds_spline_filter = Qds_spline            
    if filter_Qdds_toTrack:
        Qdds_spline_filter = filterDataFrame(
            Qdds_spline, cutoff_frequency=cutoff_freq_Qdds)
    else:
        Qdds_spline_filter = Qdds_spline        
    # Instead of splining Qs twice to get Qdds, spline (filtered) Qds.
    if splineQds:
        Qdds_spline2 = pd.DataFrame(
                columns=Qs_fromIK_sel.columns)
        Qdds_spline2['time'] = Qs_fromIK_sel['time']
        for joint in joints:
            spline = interpolate.InterpolatedUnivariateSpline(
                Qds_spline_filter['time'], Qds_spline_filter[joint], k=3)
            splineD1 = spline.derivative(n=1)
            Qdds_spline2[joint] = splineD1(Qds_spline_filter['time'])                
        if filter_Qdds_toTrack:
            Qdds_spline_filter = filterDataFrame(
                Qdds_spline2, cutoff_frequency=cutoff_freq_Qdds)
        else:
            Qdds_spline_filter = Qdds_spline2        
    # Interpolate.
    Qds_spline_interp = interpolateDataFrame(
        Qds_spline_filter, timeIntervals[0], timeIntervals[1], N)
    Qdds_spline_interp = interpolateDataFrame(
        Qdds_spline_filter, timeIntervals[0], timeIntervals[1], N)
    # Scale.
    dataToTrack_Qds_sc = scaleDataFrame(
            Qds_spline_interp, scaling['Qds'], 
            coordinates_toTrack_l).to_numpy()[:,1::].T
    dataToTrack_Qdds_sc = scaleDataFrame(
            Qdds_spline_interp, scaling['Qdds'], 
            coordinates_toTrack_l).to_numpy()[:,1::].T
    # Select.
    refData_Qds_nsc = selectFromDataFrame(
        Qds_spline_interp, joints).to_numpy()[:,1::].T            
    refData_Qdds_nsc = selectFromDataFrame(
        Qdds_spline_interp, joints).to_numpy()[:,1::].T
    # Proceeed similarly with experimental data for comparison.
    if mocap_ik_data_available:
        # Initialize dataframes.     
        Qs_mocap_spline = pd.DataFrame(columns=Qs_mocap_sel.columns)
        Qds_mocap_spline = pd.DataFrame(columns=Qs_mocap_sel.columns)
        Qdds_mocap_spline = pd.DataFrame(columns=Qs_mocap_sel.columns)
        # Set time.
        Qs_mocap_spline['time'] = Qs_mocap_sel['time']
        Qds_mocap_spline['time'] = Qs_mocap_sel['time']
        Qdds_mocap_spline['time'] = Qs_mocap_sel['time']
        # Compute values.
        for joint in joints:
            spline = interpolate.InterpolatedUnivariateSpline(
                Qs_mocap_sel['time'], 
                Qs_mocap_sel[joint], k=3)
            Qs_mocap_spline[joint] = spline(
                Qs_mocap_sel['time'])
            splineD1 = spline.derivative(n=1)
            Qds_mocap_spline[joint] = splineD1(
                Qs_mocap_sel['time'])
            splineD2 = spline.derivative(n=2)
            Qdds_mocap_spline[joint] = splineD2(
                Qs_mocap_sel['time'])        
        # Filter.
        if filter_Qds_toTrack:
            Qds_mocap_spline_filter = filterDataFrame(
                Qds_mocap_spline, cutoff_frequency=cutoff_freq_Qds)
        else:
            Qds_mocap_spline_filter = Qds_mocap_spline                
        if filter_Qdds_toTrack:
            Qdds_mocap_spline_filter = filterDataFrame(
                Qdds_mocap_spline, cutoff_frequency=cutoff_freq_Qdds)
        else:
            Qdds_mocap_spline_filter = Qdds_mocap_spline            
        # Instead of splining Qs twice to get Qdds, spline (filtered) Qds.
        if splineQds:
            Qdds_mocap_spline2 = pd.DataFrame(
                columns=Qs_mocap_sel.columns)
            Qdds_mocap_spline2['time'] = Qs_mocap_sel['time']
            for joint in joints:
                spline = interpolate.InterpolatedUnivariateSpline(
                    Qds_mocap_spline_filter['time'], 
                    Qds_mocap_spline_filter[joint], k=3)
                splineD1 = spline.derivative(n=1)
                Qdds_mocap_spline2[joint] = splineD1(
                    Qds_mocap_spline_filter['time'])                
            if filter_Qdds_toTrack:
                Qdds_mocap_spline_filter = filterDataFrame(
                    Qdds_mocap_spline2, cutoff_frequency=cutoff_freq_Qdds)
            else:
                Qdds_mocap_spline_filter = Qdds_mocap_spline2        
        # Interpolate.
        Qds_mocap_ref = interpolateDataFrame(
            Qds_mocap_spline_filter, timeIntervals[0], 
            timeIntervals[1], N).to_numpy()[:,1::].T
        Qdds_mocap_ref = interpolateDataFrame(
            Qdds_mocap_spline_filter, timeIntervals[0], 
            timeIntervals[1], N).to_numpy()[:,1::].T
            
    # %% Update bounds if coordinate constraints.
    if coordinate_constraints:
        # TODO: not sure why bounds at collocation points not updated (Antoine).
        # from utilsOpenSimAD import getColfromk
        ubQsk_vec = uw['Qs'].to_numpy().T * np.ones((1, N+1))
        lbQsk_vec = lw['Qs'].to_numpy().T * np.ones((1, N+1))
        # ubQsj_vec = uw['Qs'].to_numpy().T * np.ones((1, d*N))
        # lbQsj_vec = lw['Qs'].to_numpy().T * np.ones((1, d*N))           
        for cons in coordinate_constraints:            
            if cons == 'pelvis_ty':
                pelvis_ty_sc = scaleDataFrame(Qs_toTrack, scaling['Qs'], [cons]).to_numpy()[:,1::].T
                # If there is an offset as part of the design variables,
                # the constraint is handled as a constraint and not as a
                # bound.
                if not offset_ty:
                    ubQsk_vec[joints.index(cons),:-1] = coordinate_constraints[cons]['env_bound'] / scaling['Qs'].iloc[0][cons] + pelvis_ty_sc        
                    lbQsk_vec[joints.index(cons),:-1] = -coordinate_constraints[cons]['env_bound'] / scaling['Qs'].iloc[0][cons] + pelvis_ty_sc
                    # c_sc_j = getColfromk(pelvis_ty_sc, d, N)
                    # ubQsj_vec[joints.index(cons),:] = coordinate_constraints[cons]['env_bound'] / scaling['Qs'].iloc[0][cons] + c_sc_j
                    # lbQsj_vec[joints.index(cons),:] = -coordinate_constraints[cons]['env_bound'] / scaling['Qs'].iloc[0][cons] + c_sc_j                        
            else:            
                c_sc = scaleDataFrame(Qs_toTrack, scaling['Qs'], [cons]).to_numpy()[:,1::].T                
                ubQsk_vec[joints.index(cons),:-1] = coordinate_constraints[cons]['env_bound'] / scaling['Qs'].iloc[0][cons] + c_sc        
                lbQsk_vec[joints.index(cons),:-1] = -coordinate_constraints[cons]['env_bound'] / scaling['Qs'].iloc[0][cons] + c_sc
                # c_sc_j = getColfromk(c_sc, d, N)
                # ubQsj_vec[joints.index(cons),:] = coordinate_constraints[cons]['env_bound'] / scaling['Qs'].iloc[0][cons] + c_sc_j
                # lbQsj_vec[joints.index(cons),:] = -coordinate_constraints[cons]['env_bound'] / scaling['Qs'].iloc[0][cons] + c_sc_j
        uw['Qsk'] = ca.vec(ubQsk_vec).full()
        lw['Qsk'] = ca.vec(lbQsk_vec).full()
        # uw['Qsj'] = ca.vec(ubQsj_vec).full()
        # lw['Qsj'] = ca.vec(lbQsj_vec).full()
        
    # %% Formulate optimal control problem.
    if solveProblem:
        J = 0 # initialize cost function.
        opti = ca.Opti() # initialize opti instance.            
        # Static parameters.
        if offset_ty:
            offset = opti.variable(1) # Offset pelvis_ty.
            opti.subject_to(opti.bounded(lw['Offsetk'], offset, uw['Offsetk']))
            opti.set_initial(offset, w0['Offset'])
        else:
            offset = 0   
        # Time step.
        h = timeElapsed / N
        # States.
        if torque_driven_model:
            # Coordinate activation at mesh points.
            aCoord = opti.variable(nMuscleDrivenJoints, N+1)
            opti.subject_to(opti.bounded(lw['CoordAk'], ca.vec(aCoord), uw['CoordAk']))
            opti.set_initial(aCoord, w0['CoordA'].to_numpy().T)
            assert np.all(lw['CoordAk'] <= ca.vec(w0['CoordA'].to_numpy().T).full()), "Issue with lower bound coordinate activations"
            assert np.all(uw['CoordAk'] >= ca.vec(w0['CoordA'].to_numpy().T).full()), "Issue with upper bound coordinate activations"
            # Coordinate activation at collocation points.
            aCoord_col = opti.variable(nMuscleDrivenJoints, d*N)
            opti.subject_to(opti.bounded(lw['CoordAj'], ca.vec(aCoord_col), uw['CoordAj']))
            opti.set_initial(aCoord_col, w0['CoordAj'].to_numpy().T)
            assert np.all(lw['CoordAj'] <= ca.vec(w0['CoordAj'].to_numpy().T).full()), "Issue with lower bound coordinate activations (collocation points)"
            assert np.all(uw['CoordAj'] >= ca.vec(w0['CoordAj'].to_numpy().T).full()), "Issue with upper bound coordinate activations (collocation points)"
        else:
            # Muscle activation at mesh points.
            a = opti.variable(nMuscles, N+1)
            opti.subject_to(opti.bounded(lw['Ak'], ca.vec(a), uw['Ak']))
            opti.set_initial(a, w0['A'].to_numpy().T)
            assert np.all(lw['Ak'] <= ca.vec(w0['A'].to_numpy().T).full()), "Issue with lower bound muscle activations"
            assert np.all(uw['Ak'] >= ca.vec(w0['A'].to_numpy().T).full()), "Issue with upper bound muscle activations"
            # Muscle activation at collocation points.
            a_col = opti.variable(nMuscles, d*N)
            opti.subject_to(opti.bounded(lw['Aj'], ca.vec(a_col), uw['Aj']))
            opti.set_initial(a_col, w0['Aj'].to_numpy().T)
            assert np.all(lw['Aj'] <= ca.vec(w0['Aj'].to_numpy().T).full()), "Issue with lower bound muscle activations (collocation points)"
            assert np.all(uw['Aj'] >= ca.vec(w0['Aj'].to_numpy().T).full()), "Issue with upper bound muscle activations (collocation points)"
            # Muscle force at mesh points.
            nF = opti.variable(nMuscles, N+1)
            opti.subject_to(opti.bounded(lw['Fk'], ca.vec(nF), uw['Fk']))
            opti.set_initial(nF, w0['F'].to_numpy().T)
            assert np.all(lw['Fk'] <= ca.vec(w0['F'].to_numpy().T).full()), "Issue with lower bound muscle forces"
            assert np.all(uw['Fk'] >= ca.vec(w0['F'].to_numpy().T).full()), "Issue with upper bound muscle forces"
            # Muscle force at collocation points.
            nF_col = opti.variable(nMuscles, d*N)
            opti.subject_to(opti.bounded(lw['Fj'], ca.vec(nF_col), uw['Fj']))
            opti.set_initial(nF_col, w0['Fj'].to_numpy().T)
            assert np.all(lw['Fj'] <= ca.vec(w0['Fj'].to_numpy().T).full()), "Issue with lower bound muscle forces (collocation points)"
            assert np.all(uw['Fj'] >= ca.vec(w0['Fj'].to_numpy().T).full()), "Issue with upper bound muscle forces (collocation points)"
        # Joint position at mesh points.
        Qs = opti.variable(nJoints, N+1)
        opti.subject_to(opti.bounded(lw['Qsk'], ca.vec(Qs), uw['Qsk']))
        guessQsEnd = np.concatenate(
            (w0['Qs'].to_numpy().T, np.reshape(
                w0['Qs'].to_numpy().T[:,-1], 
                (w0['Qs'].to_numpy().T.shape[0], 1))), axis=1)
        opti.set_initial(Qs, guessQsEnd)
        # Small margin to account for filtering.
        assert np.all(lw['Qsk'] - np.pi/180 <= ca.vec(guessQsEnd).full()), "Issue with lower bound coordinate values"
        assert np.all(uw['Qsk'] + np.pi/180 >= ca.vec(guessQsEnd).full()), "Issue with upper bound coordinate values"
        # Joint position at collocation points.
        Qs_col = opti.variable(nJoints, d*N)
        opti.subject_to(opti.bounded(lw['Qsj'], ca.vec(Qs_col), uw['Qsj']))
        opti.set_initial(Qs_col, w0['Qsj'].to_numpy().T)
        # Small margin to account for filtering.
        assert np.all(lw['Qsj'] - np.pi/180 <= ca.vec(w0['Qsj'].to_numpy().T).full()), "Issue with lower bound coordinate values (collocation points)"
        assert np.all(uw['Qsj'] + np.pi/180 >= ca.vec(w0['Qsj'].to_numpy().T).full()), "Issue with upper bound coordinate values (collocation points)"
        # Joint velocity at mesh points.
        Qds = opti.variable(nJoints, N+1)
        opti.subject_to(opti.bounded(lw['Qdsk'], ca.vec(Qds), uw['Qdsk']))
        guessQdsEnd = np.concatenate(
            (w0['Qds'].to_numpy().T, np.reshape(
                w0['Qds'].to_numpy().T[:,-1], 
                (w0['Qds'].to_numpy().T.shape[0], 1))), axis=1)
        opti.set_initial(Qds, guessQdsEnd)
        assert np.all(lw['Qdsk'] <= ca.vec(guessQdsEnd).full()), "Issue with lower bound coordinate speeds"
        assert np.all(uw['Qdsk'] >= ca.vec(guessQdsEnd).full()), "Issue with upper bound coordinate speeds"        
        # Joint velocity at collocation points.
        Qds_col = opti.variable(nJoints, d*N)
        opti.subject_to(opti.bounded(lw['Qdsj'], ca.vec(Qds_col), uw['Qdsj']))
        opti.set_initial(Qds_col, w0['Qdsj'].to_numpy().T)
        assert np.all(lw['Qdsj'] <= ca.vec(w0['Qdsj'].to_numpy().T).full()), "Issue with lower bound coordinate speeds (collocation points)"
        assert np.all(uw['Qdsj'] >= ca.vec(w0['Qdsj'].to_numpy().T).full()), "Issue with upper bound coordinate speeds (collocation points)"
        if withArms:
            # Arm activation at mesh points.
            aArm = opti.variable(nArmJoints, N+1)
            opti.subject_to(opti.bounded(lw['ArmAk'], ca.vec(aArm), uw['ArmAk']))
            opti.set_initial(aArm, w0['ArmA'].to_numpy().T)
            assert np.all(lw['ArmAk'] <= ca.vec(w0['ArmA'].to_numpy().T).full()), "Issue with lower bound arm activations"
            assert np.all(uw['ArmAk'] >= ca.vec(w0['ArmA'].to_numpy().T).full()), "Issue with upper bound arm activations"
            # Arm activation at collocation points.
            aArm_col = opti.variable(nArmJoints, d*N)
            opti.subject_to(opti.bounded(lw['ArmAj'], ca.vec(aArm_col), uw['ArmAj']))
            opti.set_initial(aArm_col, w0['ArmAj'].to_numpy().T)
            assert np.all(lw['ArmAj'] <= ca.vec(w0['ArmAj'].to_numpy().T).full()), "Issue with lower bound arm activations (collocation points)"
            assert np.all(uw['ArmAj'] >= ca.vec(w0['ArmAj'].to_numpy().T).full()), "Issue with upper bound arm activations (collocation points)"
        if withLumbarCoordinateActuators:
            # Lumbar activation at mesh points.
            aLumbar = opti.variable(nLumbarJoints, N+1)
            opti.subject_to(opti.bounded(lw['LumbarAk'], ca.vec(aLumbar), uw['LumbarAk']))
            opti.set_initial(aLumbar, w0['LumbarA'].to_numpy().T)
            assert np.all(lw['LumbarAk'] <= ca.vec(w0['LumbarA'].to_numpy().T).full()), "Issue with lower bound lumbar activations"
            assert np.all(uw['LumbarAk'] >= ca.vec(w0['LumbarA'].to_numpy().T).full()), "Issue with upper bound lumbar activations"
            # Lumbar activation at collocation points.
            aLumbar_col = opti.variable(nLumbarJoints, d*N)
            opti.subject_to(opti.bounded(lw['LumbarAj'], ca.vec(aLumbar_col), uw['LumbarAj']))
            opti.set_initial(aLumbar_col, w0['LumbarAj'].to_numpy().T)
            assert np.all(lw['LumbarAj'] <= ca.vec(w0['LumbarAj'].to_numpy().T).full()), "Issue with lower bound lumbar activations (collocation points)"
            assert np.all(uw['LumbarAj'] >= ca.vec(w0['LumbarAj'].to_numpy().T).full()), "Issue with upper bound lumbar activations (collocation points)"
        # Controls.
        if torque_driven_model:
            # Coordinate excitation at mesh points.
            eCoord = opti.variable(nMuscleDrivenJoints, N)
            opti.subject_to(opti.bounded(lw['CoordEk'], ca.vec(eCoord), uw['CoordEk']))
            opti.set_initial(eCoord, w0['CoordE'].to_numpy().T)
            assert np.all(lw['CoordEk'] <= ca.vec(w0['CoordE'].to_numpy().T).full()), "Issue with lower bound coordinate excitations"
            assert np.all(uw['CoordEk'] >= ca.vec(w0['CoordE'].to_numpy().T).full()), "Issue with upper bound coordinate excitations"
        else:
            # Muscle activation derivative at mesh points.
            aDt = opti.variable(nMuscles, N)
            opti.subject_to(opti.bounded(lw['ADtk'], ca.vec(aDt), uw['ADtk']))
            opti.set_initial(aDt, w0['ADt'].to_numpy().T)
            assert np.all(lw['ADtk'] <= ca.vec(w0['ADt'].to_numpy().T).full()), "Issue with lower bound muscle activation derivatives"
            assert np.all(uw['ADtk'] >= ca.vec(w0['ADt'].to_numpy().T).full()), "Issue with upper bound muscle activation derivatives"
        if withArms:
            # Arm excitation at mesh points.
            eArm = opti.variable(nArmJoints, N)
            opti.subject_to(opti.bounded(lw['ArmEk'], ca.vec(eArm), uw['ArmEk']))
            opti.set_initial(eArm, w0['ArmE'].to_numpy().T)
            assert np.all(lw['ArmEk'] <= ca.vec(w0['ArmE'].to_numpy().T).full()), "Issue with lower bound arm excitations"
            assert np.all(uw['ArmEk'] >= ca.vec(w0['ArmE'].to_numpy().T).full()), "Issue with upper bound arm excitations"
        if withLumbarCoordinateActuators:
            # Lumbar excitation at mesh points.
            eLumbar = opti.variable(nLumbarJoints, N)
            opti.subject_to(opti.bounded(lw['LumbarEk'], ca.vec(eLumbar), uw['LumbarEk']))
            opti.set_initial(eLumbar, w0['LumbarE'].to_numpy().T)
            assert np.all(lw['LumbarEk'] <= ca.vec(w0['LumbarE'].to_numpy().T).full()), "Issue with lower bound lumbar excitations"
            assert np.all(uw['LumbarEk'] >= ca.vec(w0['LumbarE'].to_numpy().T).full()), "Issue with upper bound lumbar excitations"
        if not torque_driven_model:
            # Muscle force derivative at mesh points.
            nFDt = opti.variable(nMuscles, N)
            opti.subject_to(opti.bounded(lw['FDtk'], ca.vec(nFDt), uw['FDtk']))
            opti.set_initial(nFDt, w0['FDt'].to_numpy().T)
            assert np.all(lw['FDtk'] <= ca.vec(w0['FDt'].to_numpy().T).full()), "Issue with lower bound muscle force derivatives"
            assert np.all(uw['FDtk'] >= ca.vec(w0['FDt'].to_numpy().T).full()), "Issue with upper bound muscle force derivatives"
        # Joint velocity derivative (acceleration) at mesh points.
        Qdds = opti.variable(nJoints, N)
        opti.subject_to(opti.bounded(lw['Qddsk'], ca.vec(Qdds), uw['Qddsk']))
        opti.set_initial(Qdds, w0['Qdds'].to_numpy().T)
        assert np.all(lw['Qddsk'] <= ca.vec(w0['Qdds'].to_numpy().T).full()), "Issue with lower bound coordinate speed derivatives"
        assert np.all(uw['Qddsk'] >= ca.vec(w0['Qdds'].to_numpy().T).full()), "Issue with upper bound coordinate speed derivatives"
        # Reserve actuator at mesh points.
        if withReserveActuators:
            rAct = {}
            for c_j in reserveActuatorCoordinates:                    
                rAct[c_j] = opti.variable(1, N)
                opti.subject_to(opti.bounded(lw['rActk'][c_j], ca.vec(rAct[c_j]), uw['rActk'][c_j]))
                opti.set_initial(rAct[c_j], w0['rAct'][c_j].to_numpy().T)
                assert np.all(lw['rActk'][c_j] <= ca.vec(w0['rAct'][c_j].to_numpy().T).full()), "Issue with lower bound reserve actuators"
                assert np.all(uw['rActk'][c_j] >= ca.vec(w0['rAct'][c_j].to_numpy().T).full()), "Issue with upper bound reserve actuators"
            
        # %% Plots initial guess vs bounds.
        plotGuessVsBounds = False
        if plotGuessVsBounds: 
            from plotsOpenSimAD import plotGuessVSBounds
            plotGuessVSBounds(lw, uw, w0, nJoints, N, d, guessQsEnd, 
                              guessQdsEnd, withArms=withArms, 
                              withLumbarCoordinateActuators=
                              withLumbarCoordinateActuators,
                              torque_driven_model=torque_driven_model)
            
        # %% Unscale design variables.
        if not torque_driven_model:
            nF_nsc = nF * (scaling['F'].to_numpy().T * np.ones((1, N+1)))
            nF_col_nsc = nF_col * (scaling['F'].to_numpy().T * np.ones((1, d*N)))
            aDt_nsc = aDt * (scaling['ADt'].to_numpy().T * np.ones((1, N)))
            nFDt_nsc = nFDt * (scaling['FDt'].to_numpy().T * np.ones((1, N)))
        Qs_nsc = Qs * (scaling['Qs'].to_numpy().T * np.ones((1, N+1)))
        Qs_col_nsc = Qs_col * (scaling['Qs'].to_numpy().T * np.ones((1, d*N)))
        Qds_nsc = Qds * (scaling['Qds'].to_numpy().T * np.ones((1, N+1)))
        Qds_col_nsc = Qds_col * (scaling['Qds'].to_numpy().T * np.ones((1, d*N)))            
        Qdds_nsc = Qdds * (scaling['Qdds'].to_numpy().T * np.ones((1, N)))        
        if withReserveActuators:
            rAct_nsc = {}
            for c_j in reserveActuatorCoordinates:
                rAct_nsc[c_j] = rAct[c_j] * (scaling['rAct'][c_j].to_numpy().T * np.ones((1, N)))
                
        # %% Add offset data to pelvis_ty values to track.
        dataToTrack_Qs_nsc_offset = ca.MX(dataToTrack_Qs_nsc.shape[0],
                                          dataToTrack_Qs_nsc.shape[1])                    
        for j, joint in enumerate(coordinates_toTrack):                        
            if joint == "pelvis_ty":                        
                dataToTrack_Qs_nsc_offset[j, :] = dataToTrack_Qs_nsc[j, :] + offset
            else:
                dataToTrack_Qs_nsc_offset[j, :] = dataToTrack_Qs_nsc[j, :]
        # Scale Qs to track.
        dataToTrack_Qs_sc_offset = (
            dataToTrack_Qs_nsc_offset / 
            ((scaling['Qs'].to_numpy().T)[idx_coordinates_toTrack] * 
              np.ones((1, N))))
        
        # %%  Loop over mesh points.
        for k in range(N):
            # Variables within current mesh.
            # States.
            if torque_driven_model:
                aCoordkj = (ca.horzcat(aCoord[:, k], aCoord_col[:, k*d:(k+1)*d]))
            else:
                akj = (ca.horzcat(a[:, k], a_col[:, k*d:(k+1)*d]))
                nFkj = (ca.horzcat(nF[:, k], nF_col[:, k*d:(k+1)*d]))
                nFkj_nsc = (ca.horzcat(nF_nsc[:, k], nF_col_nsc[:, k*d:(k+1)*d]))
            Qskj = (ca.horzcat(Qs[:, k], Qs_col[:, k*d:(k+1)*d]))
            Qskj_nsc = (ca.horzcat(Qs_nsc[:, k], Qs_col_nsc[:, k*d:(k+1)*d]))
            Qdskj = (ca.horzcat(Qds[:, k], Qds_col[:, k*d:(k+1)*d]))    
            Qdskj_nsc = (ca.horzcat(Qds_nsc[:, k], Qds_col_nsc[:, k*d:(k+1)*d]))
            if withArms:
                aArmkj = (ca.horzcat(aArm[:, k], aArm_col[:, k*d:(k+1)*d]))
            if withLumbarCoordinateActuators:
                aLumbarkj = (ca.horzcat(aLumbar[:, k], aLumbar_col[:, k*d:(k+1)*d]))
            # Controls.
            if torque_driven_model:
                eCoordk = eCoord[:, k]
            else:
                aDtk = aDt[:, k]
                aDtk_nsc = aDt_nsc[:, k]
                nFDtk = nFDt[:, k]
                nFDtk_nsc = nFDt_nsc[:, k]
            Qddsk = Qdds[:, k]
            Qddsk_nsc = Qdds_nsc[:, k]
            if withArms:
                eArmk = eArm[:, k]
            if withLumbarCoordinateActuators:
                eLumbark = eLumbar[:, k]
            if withReserveActuators:
                rActk = {}
                rActk_nsc = {}
                for c_j in reserveActuatorCoordinates:
                    rActk[c_j] = rAct[c_j][:,k]
                    rActk_nsc[c_j] = rAct_nsc[c_j][:,k]  
            # Qs and Qds are intertwined in external function.
            QsQdskj_nsc = ca.MX(nJoints*2, d+1)
            QsQdskj_nsc[::2, :] = Qskj_nsc[idxJoints4F, :]
            QsQdskj_nsc[1::2, :] = Qdskj_nsc[idxJoints4F, :]         
            
            if not torque_driven_model:
                # Polynomial approximations
                # Left side.
                Qsink_l = Qskj_nsc[leftPolynomialJointIndices, 0]
                Qdsink_l = Qdskj_nsc[leftPolynomialJointIndices, 0]
                [lMTk_l, vMTk_l, dMk_l] = f_polynomial['l'](Qsink_l, Qdsink_l) 
                # Right side.
                Qsink_r = Qskj_nsc[rightPolynomialJointIndices, 0]
                Qdsink_r = Qdskj_nsc[rightPolynomialJointIndices, 0]
                [lMTk_r, vMTk_r, dMk_r] = f_polynomial['r'](Qsink_r, Qdsink_r)
                # Muscle-tendon lengths and velocities.      
                lMTk_lr = ca.vertcat(lMTk_l[leftPolynomialMuscleIndices], 
                                    lMTk_r[rightPolynomialMuscleIndices])
                vMTk_lr = ca.vertcat(vMTk_l[leftPolynomialMuscleIndices], 
                                    vMTk_r[rightPolynomialMuscleIndices])
                # Moment arms.
                dMk = {}
                # Left side.
                for joint in leftPolynomialJoints:
                    if ((joint != 'mtp_angle_l') and 
                        (joint != 'lumbar_extension') and
                        (joint != 'lumbar_bending') and 
                        (joint != 'lumbar_rotation')):
                            dMk[joint] = dMk_l[
                                momentArmIndices[joint], 
                                leftPolynomialJoints.index(joint)]
                # Right side.
                for joint in rightPolynomialJoints:
                    if ((joint != 'mtp_angle_r') and 
                        (joint != 'lumbar_extension') and
                        (joint != 'lumbar_bending') and 
                        (joint != 'lumbar_rotation')):
                            # We need to adjust momentArmIndices for the right 
                            # side since polynomial indices are 'one-sided'. 
                            # We subtract by the number of side muscles.
                            c_ma = [i - nSideMuscles for 
                                    i in momentArmIndices[joint]]
                            dMk[joint] = dMk_r[
                                c_ma, rightPolynomialJoints.index(joint)]            
                
                # Hill-equilibrium.
                [hillEquilibriumk, Fk, activeFiberForcek, passiveFiberForcek,
                normActiveFiberLengthForcek, nFiberLengthk,
                fiberVelocityk, _, _] = (f_hillEquilibrium(
                    akj[:, 0], lMTk_lr, vMTk_lr, nFkj_nsc[:, 0], nFDtk_nsc))
                 
            # Limit torques.
            passiveTorque_k = {}
            if enableLimitTorques:                    
                for joint in passiveTorqueJoints:
                    passiveTorque_k[joint] = f_passiveTorque[joint](
                        Qskj_nsc[joints.index(joint), 0], 
                        Qdskj_nsc[joints.index(joint), 0]) 
            else:
                for joint in passiveTorqueJoints:
                    passiveTorque_k[joint] = 0

            # Linear (passive) torques.
            if withMTP:
                linearPassiveTorqueMtp_k = {}
                for joint in mtpJoints:
                    linearPassiveTorqueMtp_k[joint] = (
                        f_linearPassiveMtpTorque(
                            Qskj_nsc[joints.index(joint), 0],
                            Qdskj_nsc[joints.index(joint), 0]))                    
            if withArms:
                linearPassiveTorqueArms_k = {}
                for joint in armJoints:
                    linearPassiveTorqueArms_k[joint] = (
                        f_linearPassiveArmTorque(
                            Qskj_nsc[joints.index(joint), 0],
                            Qdskj_nsc[joints.index(joint), 0]))
        
            # Call external function.
            if treadmill:
                Tk = F(ca.vertcat(
                    ca.vertcat(QsQdskj_nsc[:, 0],
                               Qddsk_nsc[idxJoints4F]),
                    -settings['treadmill_speed']))
            else:
                Tk = F(ca.vertcat(QsQdskj_nsc[:, 0], 
                                   Qddsk_nsc[idxJoints4F]))
                    
            # Loop over collocation points.
            for j in range(d):                    
                # Expression for the state derivatives.
                if torque_driven_model:
                    aCoordp = ca.mtimes(aCoordkj, C[j+1])
                else:
                    ap = ca.mtimes(akj, C[j+1])        
                    nFp_nsc = ca.mtimes(nFkj_nsc, C[j+1])
                Qsp_nsc = ca.mtimes(Qskj_nsc, C[j+1])
                Qdsp_nsc = ca.mtimes(Qdskj_nsc, C[j+1])
                if withArms:
                    aArmp = ca.mtimes(aArmkj, C[j+1])
                if withLumbarCoordinateActuators:
                    aLumbarp = ca.mtimes(aLumbarkj, C[j+1])
                
                # Append collocation equations.
                if torque_driven_model:
                    # Coordinate activation dynamics.
                    aCoordDtj = f_coordinateDynamics(
                        eCoordk, aCoordkj[:, j+1])
                    opti.subject_to(h*aCoordDtj - aCoordp == 0)
                else:
                    # Muscle activation dynamics.
                    opti.subject_to((h*aDtk_nsc - ap) == 0)
                    # Muscle contraction dynamics. 
                    opti.subject_to((h*nFDtk_nsc - nFp_nsc) / 
                                    scaling['F'].to_numpy().T == 0)
                # Skeleton dynamics.
                # Position derivative.
                opti.subject_to((h*Qdskj_nsc[:, j+1] - Qsp_nsc) / 
                                scaling['Qs'].to_numpy().T == 0)
                # Velocity derivative.
                opti.subject_to((h*Qddsk_nsc - Qdsp_nsc) / 
                                scaling['Qds'].to_numpy().T == 0)
                if withArms:
                    # Arm activation dynamics.
                    aArmDtj = f_armDynamics(
                        eArmk, aArmkj[:, j+1])
                    opti.subject_to(h*aArmDtj - aArmp == 0) 
                if withLumbarCoordinateActuators:
                    # Lumbar activation dynamics.
                    aLumbarDtj = f_lumbarDynamics(
                        eLumbark, aLumbarkj[:, j+1])
                    opti.subject_to(h*aLumbarDtj - aLumbarp == 0)
                
                # Cost function
                jointAccelerationTerm = f_nJointsSum2(Qddsk)
                positionTrackingTerm = f_NQsToTrackWSum2(
                    Qskj[idx_coordinates_toTrack, 0],
                    dataToTrack_Qs_sc_offset[:, k], w_dataToTrack)
                velocityTrackingTerm = f_NQsToTrackWSum2(
                    Qdskj[idx_coordinates_toTrack, 0],
                    dataToTrack_Qds_sc[:, k], w_dataToTrack)                
                J += ((
                    weights['positionTrackingTerm'] * positionTrackingTerm +
                    weights['velocityTrackingTerm'] * velocityTrackingTerm +
                    weights['jointAccelerationTerm'] * jointAccelerationTerm) * h * B[j + 1])                
                if torque_driven_model:
                    coordinateExcitationTerm = f_nCoordinatesSum2(eCoordk)
                    J += (weights['coordinateExcitationTerm'] * 
                          coordinateExcitationTerm * h * B[j + 1])
                else:
                    activationTerm = f_NMusclesSumWeightedPow(
                        akj[:, j+1], s_muscleVolume * w_muscles)
                    activationDtTerm = f_NMusclesSum2(aDtk)
                    forceDtTerm = f_NMusclesSum2(nFDtk)
                    J += ((
                        weights['activationTerm'] * activationTerm +                       
                        weights['activationDtTerm'] * activationDtTerm + 
                        weights['forceDtTerm'] * forceDtTerm) * h * B[j + 1])                
                if withArms:     
                    armExcitationTerm = f_nArmJointsSum2(eArmk)
                    J += (weights['armExcitationTerm'] * 
                          armExcitationTerm * h * B[j + 1])
                if withLumbarCoordinateActuators:     
                    lumbarExcitationTerm = f_nLumbarJointsSum2(eLumbark)
                    J += (weights['lumbarExcitationTerm'] * 
                          lumbarExcitationTerm * h * B[j + 1])
                if trackQdds:
                    accelerationTrackingTerm = f_NQsToTrackWSum2(
                        Qddsk[idx_coordinates_toTrack],
                        dataToTrack_Qdds_sc[:, k],
                        w_dataToTrack)
                    J += (weights['accelerationTrackingTerm'] * 
                          accelerationTrackingTerm * h * B[j + 1])                    
                if withReserveActuators:
                    reserveActuatorTerm = 0
                    for c_j in reserveActuatorCoordinates:                        
                        reserveActuatorTerm += ca.sumsqr(rActk[c_j])                            
                    reserveActuatorTerm /= len(reserveActuatorCoordinates)
                    J += (weights['reserveActuatorTerm'] * 
                          reserveActuatorTerm * h * B[j + 1])
                    
                if min_ratio_vGRF and weights['vGRFRatioTerm'] > 0:                    
                    for side in contactSides:                        
                        vGRF_ratio = ca.sqrt(
                            (ca.sum1(Tk[idx_vGRF_front[side]])) /
                            (ca.sum1(Tk[idx_vGRF_rear[side]])))
                        J += (weights['vGRFRatioTerm'] * 
                              (vGRF_ratio) * h * B[j + 1]) 

                # TODO_footVelocity make foot y velocity 0 clean and move things the right place
                # should compute velocity in F.cpp (body linear velocity)
                # also pre-compute all this stuff appropriately above
                # also probably don't assume 0 velocity at all times
                if k==0 and j== 0:
                    idx_calcn_l = Fmap['body_origins']['calcn_l']
                    idx_calcn_r = Fmap['body_origins']['calcn_r']
                    calcn_y_l_old = Tk[idx_calcn_l+1]
                    calcn_y_r_old = Tk[idx_calcn_r+1]
                    weights['calcnVelocityTerm'] = 100
                calcn_vel = ca.sqr(Tk[idx_calcn_l] - calcn_y_l_old) + ca.sqr(Tk[idx_calcn_r] - calcn_y_r_old)
                J += (weights['calcnVelocityTerm'] * 
                      calcn_vel * h * B[j + 1]) 

            # Note: we only impose the following constraints at the mesh
            # points. To be fully consistent with an orthogonal radau
            # collocation scheme, we should impose them at the collocation
            # points too. This would increase the size of the problem.
            # Null pelvis residuals (dynamic consistency).
            opti.subject_to(Tk[idxGroundPelvisJointsinF, 0] == 0)
            
            # Skeleton dynamics.
            if torque_driven_model:
                for cj, joint in enumerate(muscleDrivenJoints):                        
                    coordActk_joint = (
                        scaling['CoordE'].iloc[0][joint] * aCoordkj[cj, 0])
                    # Add contribution of reserve actuator.
                    if withReserveActuators and joint in reserveActuatorCoordinates:
                        coordActk_joint += rActk_nsc[joint]
                    diffTk_joint = f_diffTorques(
                        Tk[F_map['residuals'][joint]],
                        coordActk_joint, passiveTorque_k[joint])
                    opti.subject_to(diffTk_joint == 0)
            else:
                # Muscle-driven joint torques.
                for joint in muscleDrivenJoints:                
                    Fk_joint = Fk[momentArmIndices[joint]]
                    mTk_joint = ca.sum1(dMk[joint]*Fk_joint)
                    # Add contribution of reserve actuator.
                    if withReserveActuators and joint in reserveActuatorCoordinates:
                        mTk_joint += rActk_nsc[joint]
                    diffTk_joint = f_diffTorques(
                        Tk[F_map['residuals'][joint] ], mTk_joint,
                        passiveTorque_k[joint])
                    opti.subject_to(diffTk_joint == 0)
                
            # TODO: clean up lumbar vs arms vs MTP for consistency.
            # Torque-driven joint torques.            
            # Lumbar joints.
            if withLumbarCoordinateActuators:
                for cj, joint in enumerate(lumbarJoints):                        
                    coordAct_lumbark = (
                        scaling['LumbarE'].iloc[0][joint] * aLumbarkj[cj, 0])
                    passiveTorque_Lumbark = passiveTorque_k[joint]
                    # Add contribution of reserve actuator.
                    if withReserveActuators and joint in reserveActuatorCoordinates:
                        passiveTorque_Lumbark += rActk_nsc[joint]
                    diffTk_lumbar = f_diffTorques(
                        Tk[F_map['residuals'][joint]], coordAct_lumbark, 
                        passiveTorque_Lumbark)
                    opti.subject_to(diffTk_lumbar == 0)
            
            # Arm joints.
            # Note: there isn't really a reason to have scaled constraints.
            # Should revise in future versions.
            if withArms:
                for cj, joint in enumerate(armJoints):                    
                    passiveTorque_Armsk = linearPassiveTorqueArms_k[joint]
                    # Add contribution of reserve actuator.
                    if withReserveActuators and joint in reserveActuatorCoordinates:
                        passiveTorque_Armsk += rActk_nsc[joint]
                    diffTk_joint = f_diffTorques(
                        Tk[F_map['residuals'][joint] ] / 
                        scaling['ArmE'].iloc[0][joint],
                        aArmkj[cj, 0], passiveTorque_Armsk /
                        scaling['ArmE'].iloc[0][joint])
                    opti.subject_to(diffTk_joint == 0)
            
            # Mtp joints.
            if withMTP:
                for joint in mtpJoints:                    
                    passiveTorque_MTPk = (passiveTorque_k[joint] + 
                                          linearPassiveTorqueMtp_k[joint])
                    # Add contribution of reserve actuator.
                    if withReserveActuators and joint in reserveActuatorCoordinates:
                        passiveTorque_MTPk += rActk_nsc[joint]
                    diffTk_joint = f_diffTorques(
                        Tk[F_map['residuals'][joint]], 0, passiveTorque_MTPk)
                    opti.subject_to(diffTk_joint == 0)
            
            if not torque_driven_model:
                # Activation dynamics.
                act1 = aDtk_nsc + akj[:, 0] / deactivationTimeConstant
                act2 = aDtk_nsc + akj[:, 0] / activationTimeConstant
                opti.subject_to(act1 >= 0)
                opti.subject_to(act2 <= 1 / activationTimeConstant)
                
                # Contraction dynamics.
                opti.subject_to(hillEquilibriumk == 0)
            
            # Equality / continuity constraints.
            if torque_driven_model:
                opti.subject_to(aCoord[:, k+1] == ca.mtimes(aCoordkj, D))
            else:
                opti.subject_to(a[:, k+1] == ca.mtimes(akj, D))
                opti.subject_to(nF[:, k+1] == ca.mtimes(nFkj, D))    
            opti.subject_to(Qs[:, k+1] == ca.mtimes(Qskj, D))
            opti.subject_to(Qds[:, k+1] == ca.mtimes(Qdskj, D))    
            if withArms:
                opti.subject_to(aArm[:, k+1] == ca.mtimes(aArmkj, D))
            if withLumbarCoordinateActuators:
                opti.subject_to(aLumbar[:, k+1] == ca.mtimes(aLumbarkj, D))
                
            # Other constraints.                
            # We might want the model's heels to remain in contact with the
            # ground. We do that here by enforcing that the vertical ground
            # reaction force of the heel contact spheres is larger than 
            # heel_vGRF_threshold.
            if heel_vGRF_threshold > 0:
                vGRFk = Tk[idx_vGRF_heel]
                opti.subject_to(vGRFk > heel_vGRF_threshold)
                    
            # To prevent the feet to penetrate the ground, as might happen
            # at the beginning of the simulation, we might want to enforce 
            # that the vertical position of the origin fo the calcaneus and 
            # toes segments is above yCalcnToesThresholds.
            if yCalcnToes:
                yCalcnToesk = Tk[idx_yCalcnToes]
                opti.subject_to(yCalcnToesk > yCalcnToesThresholds)
            
        # Periodic constraints.
        if periodicConstraints:
            # Coordinate values.
            if 'coordinateValues' in periodicConstraints:
                opti.subject_to(Qs[idxPeriodicQs, -1] - 
                                Qs[idxPeriodicQs, 0] == 0)
            # Coordinate speeds.
            if 'coordinateSpeeds' in periodicConstraints:
                opti.subject_to(Qds[idxPeriodicQds, -1] - 
                                Qds[idxPeriodicQds, 0] == 0)
            # Muscle activations and forces.
            if 'muscleActivationsForces' in periodicConstraints:
                opti.subject_to(a[idxPeriodicMuscles, -1] - 
                                a[idxPeriodicMuscles, 0] == 0)
                opti.subject_to(nF[idxPeriodicMuscles, -1] - 
                                nF[idxPeriodicMuscles, 0] == 0)
                
            # Coordinate activations.
            if ('lowerLimbJointActivations' in periodicConstraints and
                torque_driven_model):
                opti.subject_to(aCoord[idxPeriodicMuscles, -1] - 
                                aCoord[idxPeriodicMuscles, 0] == 0)

            # Lumbar activations.
            if 'lumbarJointActivations' in periodicConstraints:                
                opti.subject_to(aLumbar[idxPeriodicLumbar, -1] - 
                                aLumbar[idxPeriodicLumbar, 0] == 0)
                
        # Constraints on pelvis_ty if offset as design variable.
        if offset_ty and 'pelvis_ty' in coordinate_constraints:                
            pelvis_ty_sc_offset = (
                pelvis_ty_sc + offset / 
                scaling['Qs'].iloc[0]["pelvis_ty"])            
            opti.subject_to(opti.bounded(
                -coordinate_constraints['pelvis_ty']["env_bound"] / 
                scaling['Qs'].iloc[0]["pelvis_ty"],
                Qs[joints.index("pelvis_ty"), :-1] - 
                pelvis_ty_sc_offset[0, :], 
                coordinate_constraints['pelvis_ty']["env_bound"] / 
                scaling['Qs'].iloc[0]["pelvis_ty"]))
        
        # Create NLP solver.
        opti.minimize(J)
        
        # Solve problem.
        # When using the default opti, bounds are replaced by constraints,
        # which is not what we want. This functions allows using bounds and not
        # constraints.
        from utilsOpenSimAD import solve_with_bounds
        w_opt, stats = solve_with_bounds(opti, ipopt_tolerance,
                                         useExpressionGraphFunction)             
        np.save(os.path.join(pathResults, 'w_opt_{}.npy'.format(case)), w_opt)
        np.save(os.path.join(pathResults, 'stats_{}.npy'.format(case)), stats)
        
    # %% Analyze results.
    if analyzeResults:
        w_opt = np.load(os.path.join(pathResults, 'w_opt_{}.npy'.format(case)))
        stats = np.load(os.path.join(pathResults, 'stats_{}.npy'.format(case)), 
                        allow_pickle=True).item()  
        if not stats['success'] == True:
            print('PROBLEM DID NOT CONVERGE - {} - {} - {} \n\n'.format( 
                  stats['return_status'], subject, trialName))
            return
        
        # Extract results.
        starti = 0
        if offset_ty:
            offset_opt = w_opt[starti:starti+1]
            starti += 1
        if torque_driven_model:
            aCoord_opt = (
                np.reshape(w_opt[starti:starti+nMuscleDrivenJoints*(N+1)],
                           (N+1, nMuscleDrivenJoints))).T
            starti = starti + nMuscleDrivenJoints*(N+1)    
            aCoord_col_opt = (
                np.reshape(w_opt[starti:starti+nMuscleDrivenJoints*(d*N)],
                           (d*N, nMuscleDrivenJoints))).T
            starti = starti + nMuscleDrivenJoints*(d*N)
        else:
            a_opt = (
                np.reshape(w_opt[starti:starti+nMuscles*(N+1)], (N+1, nMuscles))).T
            starti = starti + nMuscles*(N+1)
            a_col_opt = (
                np.reshape(w_opt[starti:starti+nMuscles*(d*N)], (d*N, nMuscles))).T    
            starti = starti + nMuscles*(d*N)
            nF_opt = (
                np.reshape(w_opt[starti:starti+nMuscles*(N+1)], (N+1, nMuscles))).T  
            starti = starti + nMuscles*(N+1)
            nF_col_opt = (
                np.reshape(w_opt[starti:starti+nMuscles*(d*N)], (d*N, nMuscles))).T
            starti = starti + nMuscles*(d*N)
        Qs_opt = (
            np.reshape(w_opt[starti:starti+nJoints*(N+1)], (N+1, nJoints))  ).T  
        starti = starti + nJoints*(N+1)    
        Qs_col_opt = (
            np.reshape(w_opt[starti:starti+nJoints*(d*N)], (d*N, nJoints))).T
        starti = starti + nJoints*(d*N)
        Qds_opt = (
            np.reshape(w_opt[starti:starti+nJoints*(N+1)], (N+1, nJoints)) ).T   
        starti = starti + nJoints*(N+1)    
        Qds_col_opt = (
            np.reshape(w_opt[starti:starti+nJoints*(d*N)], (d*N, nJoints))).T
        starti = starti + nJoints*(d*N)    
        if withArms:
            aArm_opt = (
                np.reshape(w_opt[starti:starti+nArmJoints*(N+1)],
                           (N+1, nArmJoints))).T
            starti = starti + nArmJoints*(N+1)    
            aArm_col_opt = (
                np.reshape(w_opt[starti:starti+nArmJoints*(d*N)],
                           (d*N, nArmJoints))).T
            starti = starti + nArmJoints*(d*N)
        if withLumbarCoordinateActuators:
            aLumbar_opt = (
                np.reshape(w_opt[starti:starti+nLumbarJoints*(N+1)],
                           (N+1, nLumbarJoints))).T
            starti = starti + nLumbarJoints*(N+1)    
            aLumbar_col_opt = (
                np.reshape(w_opt[starti:starti+nLumbarJoints*(d*N)],
                           (d*N, nLumbarJoints))).T
            starti = starti + nLumbarJoints*(d*N)
        if torque_driven_model:
            eCoord_opt = (
                np.reshape(w_opt[starti:starti+nMuscleDrivenJoints*N],
                           (N, nMuscleDrivenJoints))).T
            starti = starti + nMuscleDrivenJoints*N
        else:
            aDt_opt = (
                np.reshape(w_opt[starti:starti+nMuscles*N], (N, nMuscles))).T
            starti = starti + nMuscles*N 
        if withArms:
            eArm_opt = (
                np.reshape(w_opt[starti:starti+nArmJoints*N],
                           (N, nArmJoints))).T
            starti = starti + nArmJoints*N
        if withLumbarCoordinateActuators:
            eLumbar_opt = (
                np.reshape(w_opt[starti:starti+nLumbarJoints*N],
                           (N, nLumbarJoints))).T
            starti = starti + nLumbarJoints*N
        if not torque_driven_model:
            nFDt_opt = (
                np.reshape(w_opt[starti:starti+nMuscles*(N)], (N, nMuscles))).T
            starti = starti + nMuscles*(N)
        Qdds_opt = (
            np.reshape(w_opt[starti:starti+nJoints*(N)],(N, nJoints))).T
        starti = starti + nJoints*(N)
        if withReserveActuators:
            rAct_opt = {}
            for c_j in reserveActuatorCoordinates:
                rAct_opt[c_j] = (
                    np.reshape(w_opt[starti:starti+1*(N)], (N, 1))).T
                starti = starti + 1*(N)
        assert (starti == w_opt.shape[0]), "error when extracting results"
        
        # %% Visualize results against bounds.
        visualizeResultsBounds = False
        if visualizeResultsBounds:
            from plotsOpenSimAD import plotOptimalSolutionVSBounds
            c_wopt = {'Qs_opt': Qs_opt, 'Qs_col_opt': Qs_col_opt,
                      'Qds_opt': Qds_opt, 'Qds_col_opt': Qds_col_opt,
                      'Qdds_opt': Qdds_opt}
            if torque_driven_model:                
                c_wopt['aCoord_opt'] = aCoord_opt
                c_wopt['aCoord_col_opt'] = aCoord_col_opt
                c_wopt['eCoord_opt'] = eCoord_opt                
            else:
                c_wopt['a_opt'] = a_opt
                c_wopt['a_col_opt'] = a_col_opt
                c_wopt['nF_opt'] = nF_opt
                c_wopt['nF_col_opt'] = nF_col_opt
                c_wopt['aDt_opt'] = aDt_opt
                c_wopt['nFDt_opt'] = nFDt_opt
            plotOptimalSolutionVSBounds(lw, uw, c_wopt, 
                                        torque_driven_model=torque_driven_model)
            
        # %% Unscale results. 
        Qs_opt_nsc = Qs_opt * (
            scaling['Qs'].to_numpy().T * np.ones((1, N+1)))
        Qds_opt_nsc = Qds_opt * (
            scaling['Qds'].to_numpy().T * np.ones((1, N+1)))
        Qdds_opt_nsc = Qdds_opt * (
            scaling['Qdds'].to_numpy().T * np.ones((1, N)))
        if torque_driven_model:
            aCoord_opt_nsc = aCoord_opt * (
                scaling['CoordA'].to_numpy().T * np.ones((1, N+1)))
        else:
            nFDt_opt_nsc = nFDt_opt * (
                scaling['FDt'].to_numpy().T * np.ones((1, N)))
        if withReserveActuators:
            rAct_opt_nsc = {}
            for c_j in reserveActuatorCoordinates:
                rAct_opt_nsc[c_j] = rAct_opt[c_j] * (
                    scaling['rAct'][c_j].to_numpy().T * np.ones((1, N)))
        if offset_ty:
            offset_opt_nsc = offset_opt * scaling['Offset']
        
        # %% Extract passive joint torques.
        if withMTP:           
            linearPassiveTorqueMtp_opt = np.zeros((nMtpJoints, N+1))
            passiveTorqueMtp_opt = np.zeros((nMtpJoints, N+1))
            for k in range(N+1):                    
                for cj, joint in enumerate(mtpJoints):
                    linearPassiveTorqueMtp_opt[cj, k] = (
                        f_linearPassiveMtpTorque(
                            Qs_opt_nsc[joints.index(joint), k],
                            Qds_opt_nsc[joints.index(joint), k]))
                    if enableLimitTorques:
                        passiveTorqueMtp_opt[cj, k] = (
                            f_passiveTorque[joint](
                                Qs_opt_nsc[joints.index(joint),k], 
                                Qds_opt_nsc[joints.index(joint),k]))                        
        if withArms:
            linearPassiveTorqueArms_opt = np.zeros((nArmJoints, N+1))
            for k in range(N+1):  
                for cj, joint in enumerate(armJoints):
                    linearPassiveTorqueArms_opt[cj, k] = (
                        f_linearPassiveArmTorque(
                            Qs_opt_nsc[joints.index(joint), k],
                            Qds_opt_nsc[joints.index(joint), k]))
            
        # %% Extract joint torques and ground reaction forces.
        # Helper indices
        idxGR, idxGR['GRF'], idxGR['GRF']['all']  = {}, {}, {}
        idxGR['COP'], idxGR['GRM'], idxGR['GRM']['all'] = {}, {}, {}
        for sphere in contactSpheres['all']:
            idxGR['GRF'][sphere] = {}
            idxGR['COP'][sphere] = {}
        for c_side, side in enumerate(contactSides):
            idxGR['GRF']['all'][side] = list(F_map['GRFs'][side])
            idxGR['GRM']['all'][side] = list(F_map['GRMs'][side])            
            for c_sphere, sphere in enumerate(contactSpheres[side]):                
                idxGR['GRF'][sphere][side] = list(F_map['GRFs'][sphere])                
                idxGR['COP'][sphere][side] = list(F_map['COPs'][sphere])        
        
        from utilsOpenSimAD import getCOP
        QsQds_opt_nsc = np.zeros((nJoints*2, N+1))
        QsQds_opt_nsc[::2, :] = Qs_opt_nsc[idxJoints4F, :]
        QsQds_opt_nsc[1::2, :] = Qds_opt_nsc[idxJoints4F, :]
        Qdds_opt_nsc_4F = Qdds_opt_nsc[idxJoints4F, :]
        if treadmill:
            Tj_temp = F(ca.vertcat(
                ca.vertcat(QsQds_opt_nsc[:, 0], Qdds_opt_nsc_4F[:, 0]), 
                -settings['treadmill_speed']))
        else:
            Tj_temp = F(ca.vertcat(QsQds_opt_nsc[:, 0], Qdds_opt_nsc_4F[:, 0]))          
        F_out_pp = np.zeros((Tj_temp.shape[0], N))
        if withMTP:
            mtpT = np.zeros((nMtpJoints, N))
        if withArms:
            armT = np.zeros((nArmJoints, N))
        for k in range(N):
            if treadmill:
                Tk = F(ca.vertcat(
                    ca.vertcat(QsQds_opt_nsc[:, k], Qdds_opt_nsc_4F[:, k]), 
                    -settings['treadmill_speed']))
            else:
                Tk = F(ca.vertcat(QsQds_opt_nsc[:, k], Qdds_opt_nsc_4F[:, k]))
            F_out_pp[:, k] = Tk.full().T
            if withArms:
                for cj, joint in enumerate(armJoints):
                    armT[cj, k] = f_diffTorques(
                        F_out_pp[F_map['residuals'][joint], k] / 
                        scaling['ArmE'].iloc[0][joint], 
                        aArm_opt[cj, k], 
                        linearPassiveTorqueArms_opt[cj, k] / 
                        scaling['ArmE'].iloc[0][joint])                
        # Sanity checks.
        if stats['success'] and withArms:
            assert np.all(np.abs(armT) < 10**(-ipopt_tolerance)), (
                "Error arm torques balance")                    
        if stats['success'] and withMTP:
            assert np.all(np.abs(mtpT) < 10**(-ipopt_tolerance)), (
                "Error mtp torques balance")
        # Extract GRFs, GRMs, and compute free moments and COPs. 
        GRF_all_opt, GRM_all_opt, COP_all_opt, freeT_all_opt = {}, {}, {}, {}
        GRF_s_opt, COP_s_opt = {}, {}
        GRF_all_opt['all'] = np.zeros((len(contactSides)*3, N))
        GRM_all_opt['all'] = np.zeros((len(contactSides)*3, N))
        COP_all_opt['all'] = np.zeros((len(contactSides)*3, N))
        freeT_all_opt['all'] = np.zeros((len(contactSides)*3, N))
        for c_s, side in enumerate(contactSides):
            GRF_all_opt[side] = F_out_pp[idxGR['GRF']['all'][side], :]
            GRM_all_opt[side] = F_out_pp[idxGR['GRM']['all'][side], :]
            COP_all_opt[side], freeT_all_opt[side] = getCOP(
                GRF_all_opt[side], GRM_all_opt[side])
            GRF_all_opt['all'][c_s*3:(c_s+1)*3, :] = GRF_all_opt[side]
            GRM_all_opt['all'][c_s*3:(c_s+1)*3, :] = GRM_all_opt[side]
            COP_all_opt['all'][c_s*3:(c_s+1)*3, :] = COP_all_opt[side]
            freeT_all_opt['all'][c_s*3:(c_s+1)*3, :] = freeT_all_opt[side]
            GRF_s_opt[side], COP_s_opt[side] = {}, {}
            for c_sphere, sphere in enumerate(contactSpheres[side]):                
                GRF_s_opt[side][sphere] = (
                    F_out_pp[idxGR['GRF'][sphere][side], :])
                COP_s_opt[side][sphere] = (
                    F_out_pp[idxGR['COP'][sphere][side], :])                
        # Extract joint torques.            
        torques_opt = F_out_pp[
            [F_map['residuals'][joint] for joint in joints], :]
            
        # %% Write files for visualization in OpenSim GUI.
        # Convert to degrees.
        Qs_opt_nsc_deg = np.copy(Qs_opt_nsc)
        Qs_opt_nsc_deg[idxRotationalJoints, :] = (
            Qs_opt_nsc_deg[idxRotationalJoints, :] * 180 / np.pi)
        # Labels
        GR_labels, GR_labels['GRF'] = {}, {}
        GR_labels['COP'], GR_labels['GRM'] = {}, {}
        dimensions = ['x', 'y', 'z']
        GR_labels['GRF']['all'] = {}
        GR_labels['COP']['all'] = {}
        GR_labels['GRM']['all'] = {}
        GRF_labels_fig, GRM_labels_fig, COP_labels_fig = [], [], []
        for c_side, side in enumerate(contactSides):            
            GR_labels['GRF']['all'][side] = []
            GR_labels['COP']['all'][side] = []
            GR_labels['GRM']['all'][side] = []
            GR_labels['GRF'][side] = {}
            GR_labels['COP'][side] = {}
            GR_labels['GRM'][side] = {}            
            for c_sphere, sphere in enumerate(contactSpheres[side]):                
                GR_labels['GRF'][side][sphere] = []
                GR_labels['COP'][side][sphere] = []
                GR_labels['GRM'][side][sphere] = []                
                for dimension in dimensions:                    
                    GR_labels['GRF'][side][sphere].append("ground_force_{}_v{}".format(sphere, dimension))
                    GR_labels['COP'][side][sphere].append("ground_force_{}_p{}".format(sphere, dimension))
                    GR_labels['GRM'][side][sphere].append("ground_torque_{}_{}".format(sphere, dimension))
            for dimension in dimensions:                    
                GR_labels['GRF']['all'][side].append("ground_force_{}_v{}".format(side, dimension))
                GR_labels['COP']['all'][side].append("ground_force_{}_p{}".format(side, dimension))
                GR_labels['GRM']['all'][side].append("ground_torque_{}_{}".format(side, dimension))
            GRF_labels_fig.append(GR_labels['GRF']['all'][side])
            GRM_labels_fig.append(GR_labels['GRM']['all'][side])
            COP_labels_fig.append(GR_labels['COP']['all'][side])
        GRF_labels_fig = [item for sublist in GRF_labels_fig for item in sublist]
        GRM_labels_fig = [item for sublist in GRM_labels_fig for item in sublist]
        COP_labels_fig = [item for sublist in COP_labels_fig for item in sublist]
        
        if writeGUI:
            # Kinematics and activations.
            from utils import numpy_to_storage
            labels = ['time'] + joints 
            if torque_driven_model:
                coordLabels = ([joint + '/activation' 
                                for joint in muscleDrivenJoints])
                labels += coordLabels
                
                data = np.concatenate((tgridf.T, Qs_opt_nsc_deg.T,
                                    aCoord_opt_nsc.T), axis=1)
            else:
                muscleLabels = ([bothSidesMuscle + '/activation' 
                                for bothSidesMuscle in bothSidesMuscles])
                labels += muscleLabels
                
                data = np.concatenate((tgridf.T, Qs_opt_nsc_deg.T,
                                    a_opt.T), axis=1)           
            numpy_to_storage(labels, data, os.path.join(
                pathResults, 'kinematics_activations_{}_{}.mot'.format(
                    trialName, case)), datatype='IK')
            # Torques
            labels = []
            for joint in joints:
                if (joint == 'pelvis_tx' or joint == 'pelvis_ty' or 
                    joint == 'pelvis_tz'):
                    temp_suffix = "_force"
                else:
                    temp_suffix = "_moment"
                labels.append(joint + temp_suffix)
            labels = ['time'] + labels
            data = np.concatenate((tgridf.T[:-1], torques_opt.T), axis=1) 
            numpy_to_storage(labels, data, os.path.join(
                pathResults, 'kinetics_{}_{}.mot'.format(trialName, case)),
                datatype='ID')
            # Grounds reaction forces (per sphere).
            labels = ['time']                    
            data = np.zeros((tgridf.T[:-1].shape[0], 1+nContactSpheres*9))
            data[:,0] = tgridf.T[:-1].flatten()
            idx_acc = 1
            for c_side, side in enumerate(contactSides):
                for c_sphere, sphere in enumerate(contactSpheres[side]):
                    data[:,idx_acc:idx_acc+3] = GRF_s_opt[side][sphere].T
                    idx_acc += 3
                    data[:,idx_acc:idx_acc+3] = COP_s_opt[side][sphere].T
                    idx_acc += 6
                    labels += GR_labels['GRF'][side][sphere]
                    labels += GR_labels['COP'][side][sphere]
                    labels += GR_labels['GRM'][side][sphere]
            numpy_to_storage(labels, data, os.path.join(
                pathResults, 'GRF_{}_{}.mot'.format(trialName, case)),
                datatype='GRF')
            # Grounds reaction forces (resultant).
            labels = ['time']
            for c_side, side in enumerate(contactSides):
                labels += GR_labels['GRF']['all'][side]
                labels += GR_labels['COP']['all'][side]
                labels += GR_labels['GRM']['all'][side]                
            data = np.zeros((tgridf.T[:-1].shape[0], 1+len(contactSides)*9))
            data[:,0] = tgridf.T[:-1].flatten()
            idx_acc = 1
            for c_side, side in enumerate(contactSides):
                data[:,idx_acc:idx_acc+3] = GRF_all_opt[side].T
                idx_acc += 3
                data[:,idx_acc:idx_acc+3] = COP_all_opt[side].T
                idx_acc += 3
                data[:,idx_acc:idx_acc+3] = freeT_all_opt[side].T
                idx_acc += 3     
            numpy_to_storage(labels, data, os.path.join(
                pathResults, 'GRF_resultant_{}_{}.mot'.format(
                    trialName, case)), datatype='GRF')

        # %% Data processing.
        # Reference Qs adjusted with optimized offset.
        refData_nsc = Qs_toTrack.to_numpy()[:,1::].T
        refData_offset_nsc = np.copy(refData_nsc)
        if offset_ty:                    
            refData_offset_nsc[joints.index("pelvis_ty")] = (
                refData_nsc[joints.index("pelvis_ty")] + offset_opt_nsc)         
        # Qs to track adjusted with optimized offset.
        dataToTrack_Qs_sc_offset_opt = np.zeros(
            (dataToTrack_Qs_nsc.shape[0],
             dataToTrack_Qs_nsc.shape[1]))                    
        for j, joint in enumerate(coordinates_toTrack):                        
            if joint == "pelvis_ty":                        
                dataToTrack_Qs_sc_offset_opt[j, :] = (
                    dataToTrack_Qs_nsc[j, :] + offset_opt[0][0])
            else:
                dataToTrack_Qs_sc_offset_opt[j, :] = dataToTrack_Qs_nsc[j, :]       
        dataToTrack_Qs_sc_offset_opt = (
            dataToTrack_Qs_sc_offset_opt / 
            ((scaling['Qs'].to_numpy().T)[idx_coordinates_toTrack] * 
              np.ones((1, N))))
                          
        # %% Contribution different terms to the cost function.
        # This also serves as a sanity check.
        if torque_driven_model:
            coordExcitationTerm_opt_all = 0
        else:
            activationTerm_opt_all = 0
            activationDtTerm_opt_all = 0
            forceDtTerm_opt_all = 0
        if withArms:
            armExcitationTerm_opt_all = 0
        if withLumbarCoordinateActuators:    
            lumbarExcitationTerm_opt_all = 0
        if trackQdds:
            accelerationTrackingTerm_opt_all = 0                
        jointAccelerationTerm_opt_all = 0        
        positionTrackingTerm_opt_all = 0
        velocityTrackingTerm_opt_all = 0
        if withReserveActuators:    
            reserveActuatorTerm_opt_all = 0
        if min_ratio_vGRF and weights['vGRFRatioTerm'] > 0:
            vGRFRatioTerm_opt_all = 0
        #TODO_footVelocity
        if weights['footVelocityTerm'] > 0:
            footVelTerm_opt_all = 0
        calcnVelTerm_opt_all = 0
        if not torque_driven_model:
            pMT_opt = np.zeros((len(muscleDrivenJoints), N))
            aMT_opt = np.zeros((len(muscleDrivenJoints), N))
            Ft_opt = np.zeros((nMuscles, N))
        pT_opt = np.zeros((nPassiveTorqueJoints, N))                    
        h = timeElapsed / N           
        for k in range(N):
            # States.
            if not torque_driven_model:
                akj_opt = ca.horzcat(a_opt[:, k], a_col_opt[:, k*d:(k+1)*d])
                nFkj_opt = ca.horzcat(nF_opt[:, k], nF_col_opt[:, k*d:(k+1)*d])
                nFkj_opt_nsc = nFkj_opt * (
                    scaling['F'].to_numpy().T * np.ones((1, d+1)))   
            Qskj_opt = (ca.horzcat(Qs_opt[:, k], Qs_col_opt[:, k*d:(k+1)*d]))
            Qskj_opt_nsc = Qskj_opt * (
                scaling['Qs'].to_numpy().T * np.ones((1, d+1)))
            Qdskj_opt = ca.horzcat(Qds_opt[:, k], Qds_col_opt[:, k*d:(k+1)*d])
            Qdskj_opt_nsc = Qdskj_opt * (
                scaling['Qds'].to_numpy().T * np.ones((1, d+1)))
            # Controls.
            if torque_driven_model:
                eCoordk_opt = eCoord_opt[:, k]
            else:
                aDtk_opt = aDt_opt[:, k]
                nFDtk_opt = nFDt_opt[:, k] 
                nFDtk_opt_nsc = nFDt_opt_nsc[:, k]
            if withArms:
                eArmk_opt = eArm_opt[:, k]
            if withLumbarCoordinateActuators:
                eLumbark_opt = eLumbar_opt[:, k]
            if withReserveActuators:
                rActk_opt = {}
                for c_j in reserveActuatorCoordinates:
                    rActk_opt[c_j] = rAct_opt[c_j][:, k] 
            Qddsk_opt = Qdds_opt[:, k]            
            # Joint positions and velocities are intertwined.
            QsQdskj_opt_nsc = ca.DM(nJoints*2, d+1)
            QsQdskj_opt_nsc[::2, :] = Qskj_opt_nsc
            QsQdskj_opt_nsc[1::2, :] = Qdskj_opt_nsc
            
            if not torque_driven_model:
                # Polynomial approximations.             
                # Left side.
                Qsink_opt_l = Qskj_opt_nsc[leftPolynomialJointIndices, 0]
                Qdsink_opt_l = Qdskj_opt_nsc[leftPolynomialJointIndices, 0]
                [lMTk_opt_l, vMTk_opt_l, dMk_opt_l] = f_polynomial['l'](
                    Qsink_opt_l, Qdsink_opt_l) 
                # Right side.
                Qsink_opt_r = Qskj_opt_nsc[rightPolynomialJointIndices, 0]
                Qdsink_opt_r = Qdskj_opt_nsc[rightPolynomialJointIndices, 0]
                [lMTk_opt_r, vMTk_opt_r, dMk_opt_r] = f_polynomial['r'](
                    Qsink_opt_r, Qdsink_opt_r)
                # Muscle-tendon lengths and velocities.      
                lMTk_opt_lr = ca.vertcat(
                    lMTk_opt_l[leftPolynomialMuscleIndices], 
                    lMTk_opt_r[rightPolynomialMuscleIndices])
                vMTk_opt_lr = ca.vertcat(
                    vMTk_opt_l[leftPolynomialMuscleIndices], 
                    vMTk_opt_r[rightPolynomialMuscleIndices])                
                # Moment arms.
                dMk_opt = {}
                # Left side.
                for joint in leftPolynomialJoints:
                    if ((joint != 'mtp_angle_l') and 
                        (joint != 'lumbar_extension') and
                        (joint != 'lumbar_bending') and 
                        (joint != 'lumbar_rotation')):
                            dMk_opt[joint] = dMk_opt_l[
                                momentArmIndices[joint], 
                                leftPolynomialJoints.index(joint)]
                # Right side.
                for joint in rightPolynomialJoints:
                    if ((joint != 'mtp_angle_r') and 
                        (joint != 'lumbar_extension') and
                        (joint != 'lumbar_bending') and 
                        (joint != 'lumbar_rotation')):
                            # We need to adjust momentArmIndices for the right 
                            # side since polynomial indices are 'one-sided'. 
                            # We subtract by the number of side muscles.
                            c_ma = [i - nSideMuscles for 
                                    i in momentArmIndices[joint]]
                            dMk_opt[joint] = dMk_opt_r[
                                c_ma, rightPolynomialJoints.index(joint)]
                # Hill-equilibrium.
                [hillEqk_opt, Fk_opt, _, _,_, _, _, aFPk_opt, pFPk_opt] = (
                    f_hillEquilibrium(akj_opt[:, 0], lMTk_opt_lr, vMTk_opt_lr,
                                    nFkj_opt_nsc[:, 0], nFDtk_opt_nsc))
                Ft_opt[:,k] = Fk_opt.full().flatten()   
                # Passive muscle moments.
                for c_j, joint in enumerate(muscleDrivenJoints):
                    pFk_opt_joint = pFPk_opt[momentArmIndices[joint]]
                    pMT_opt[c_j, k] = ca.sum1(dMk_opt[joint]*pFk_opt_joint)
                # Active muscle moments.
                for c_j, joint in enumerate(muscleDrivenJoints):
                    aFk_opt_joint = aFPk_opt[momentArmIndices[joint]]
                    aMT_opt[c_j, k] = ca.sum1(dMk_opt[joint]*aFk_opt_joint)    
            # Passive limit moments.
            if enableLimitTorques:
                for c_j, joint in enumerate(passiveTorqueJoints):
                    pT_opt[c_j, k] = f_passiveTorque[joint](
                        Qskj_opt_nsc[joints.index(joint), 0], 
                        Qdskj_opt_nsc[joints.index(joint), 0])
            for j in range(d):
                if torque_driven_model:
                    coordExcitationTerm_opt = f_nCoordinatesSum2(eCoordk_opt) 
                    coordExcitationTerm_opt_all += weights['coordinateExcitationTerm'] * coordExcitationTerm_opt * h * B[j + 1]
                else:
                    activationTerm_opt = f_NMusclesSumWeightedPow(akj_opt[:, j+1], s_muscleVolume * w_muscles)
                    activationDtTerm_opt = f_NMusclesSum2(aDtk_opt)
                    forceDtTerm_opt = f_NMusclesSum2(nFDtk_opt)
                    activationTerm_opt_all += weights['activationTerm'] * activationTerm_opt * h * B[j + 1]
                    activationDtTerm_opt_all += weights['activationDtTerm'] * activationDtTerm_opt * h * B[j + 1]
                    forceDtTerm_opt_all += weights['forceDtTerm'] * forceDtTerm_opt * h * B[j + 1]
                jointAccelerationTerm_opt = f_nJointsSum2(Qddsk_opt)                
                positionTrackingTerm_opt = f_NQsToTrackWSum2(Qskj_opt[idx_coordinates_toTrack, 0], dataToTrack_Qs_sc_offset_opt[:, k], w_dataToTrack)                
                velocityTrackingTerm_opt = f_NQsToTrackWSum2(Qdskj_opt[idx_coordinates_toTrack, 0], dataToTrack_Qds_sc[:, k], w_dataToTrack)                    
                positionTrackingTerm_opt_all += weights['positionTrackingTerm'] * positionTrackingTerm_opt * h * B[j + 1]
                velocityTrackingTerm_opt_all += weights['velocityTrackingTerm'] * velocityTrackingTerm_opt * h * B[j + 1]
                jointAccelerationTerm_opt_all += weights['jointAccelerationTerm'] * jointAccelerationTerm_opt * h * B[j + 1]                
                if withArms:
                    armExcitationTerm_opt = f_nArmJointsSum2(eArmk_opt) 
                    armExcitationTerm_opt_all += weights['armExcitationTerm'] * armExcitationTerm_opt * h * B[j + 1]
                if withLumbarCoordinateActuators:
                    lumbarExcitationTerm_opt = f_nLumbarJointsSum2(eLumbark_opt) 
                    lumbarExcitationTerm_opt_all += weights['lumbarExcitationTerm'] * lumbarExcitationTerm_opt * h * B[j + 1]
                if trackQdds:
                    accelerationTrackingTerm_opt = f_NQsToTrackWSum2(Qddsk_opt[idx_coordinates_toTrack], dataToTrack_Qdds_sc[:, k], w_dataToTrack)
                    accelerationTrackingTerm_opt_all += (weights['accelerationTrackingTerm'] * accelerationTrackingTerm_opt * h * B[j + 1])
                if withReserveActuators:
                    reserveActuatorTerm_opt = 0
                    for c_j in reserveActuatorCoordinates:                        
                        reserveActuatorTerm_opt += ca.sumsqr(rActk_opt[c_j])                            
                    reserveActuatorTerm_opt /= len(reserveActuatorCoordinates)
                    reserveActuatorTerm_opt_all += (weights['reserveActuatorTerm'] * reserveActuatorTerm_opt * h * B[j + 1])
                if min_ratio_vGRF and weights['vGRFRatioTerm'] > 0:
                    for side in contactSides:                    
                        vGRF_front_opt = 0                        
                        for idx_front_sphere in idx_front_spheres:
                            vGRF_front_opt += GRF_s_opt[side][contactSpheres[side][idx_front_sphere]][1,k]
                        vGRF_rear_opt = 0
                        for idx_rear_sphere in idx_rear_spheres:
                            vGRF_rear_opt += GRF_s_opt[side][contactSpheres[side][idx_rear_sphere]][1,k]
                        vGRF_ratio_opt = np.sqrt(vGRF_front_opt/vGRF_rear_opt) 
                        vGRFRatioTerm_opt_all += (weights['vGRFRatioTerm'] * vGRF_ratio_opt * h * B[j + 1])
                # TODO_footVelocity clean this
                if weights['calcnVelocityTerm'] > 0:
                    calcn_vel = ca.sqr(Tk[idx_calcn_l] - calcn_y_l_old) + ca.sqr(Tk[idx_calcn_r] - calcn_y_r_old)
                    calcnVelTerm_opt_all += (weights['calcnVelocityTerm'] * 
                        calcn_vel * h * B[j + 1]) 
                
        # "Motor control" terms.
        if torque_driven_model:
            JMotor_opt = coordExcitationTerm_opt_all.full()
        else:
            JMotor_opt = (activationTerm_opt_all.full() + 
                        activationDtTerm_opt_all.full() + 
                        forceDtTerm_opt_all.full())
        JMotor_opt += jointAccelerationTerm_opt_all.full()
        if withArms:                
            JMotor_opt += armExcitationTerm_opt_all.full()
        if withLumbarCoordinateActuators:
            JMotor_opt += lumbarExcitationTerm_opt_all.full()
        if min_ratio_vGRF and weights['vGRFRatioTerm'] > 0:
            JMotor_opt += vGRFRatioTerm_opt_all
        if withReserveActuators:    
            JMotor_opt += reserveActuatorTerm_opt_all
        # TODO_footVelocity
        if weights['calcnVelocityTerm'] > 0:
            JMotor_opt += calcnVelTerm_opt_all

        # Tracking terms.
        JTrack_opt = (positionTrackingTerm_opt_all.full() +  
                      velocityTrackingTerm_opt_all.full())
        if trackQdds:
            JTrack_opt += accelerationTrackingTerm_opt_all.full()
        # Combined terms.
        JAll_opt = JTrack_opt + JMotor_opt
        if stats['success']:
            assert np.all(
                np.abs(JAll_opt[0][0] - stats['iterations']['obj'][-1]) 
                <= 1e-5), "Error reconstructing optimal cost value"        
        JTerms = {}
        if torque_driven_model:
            JTerms["coordinateExcitationTerm"] = coordExcitationTerm_opt_all.full()[0][0]
        else:
            JTerms["activationTerm"] = activationTerm_opt_all.full()[0][0]
            JTerms["activationDtTerm"] = activationDtTerm_opt_all.full()[0][0]
            JTerms["forceDtTerm"] = forceDtTerm_opt_all.full()[0][0]
        if withArms:
            JTerms["armExcitationTerm"] = armExcitationTerm_opt_all.full()[0][0]
        if withLumbarCoordinateActuators:
            JTerms["lumbarExcitationTerm"] = lumbarExcitationTerm_opt_all.full()[0][0]
        JTerms["jointAccelerationTerm"] = jointAccelerationTerm_opt_all.full()[0][0]        
        JTerms["positionTerm"] = positionTrackingTerm_opt_all.full()[0][0]
        JTerms["velocityTerm"] = velocityTrackingTerm_opt_all.full()[0][0]
        if trackQdds:
            JTerms["accelerationTerm"] = accelerationTrackingTerm_opt_all.full()[0][0]        
        if torque_driven_model:
            JTerms["coordinateExcitationTerm_sc"] = JTerms["coordinateExcitationTerm"] / JAll_opt[0][0]
        else:
            JTerms["activationTerm_sc"] = JTerms["activationTerm"] / JAll_opt[0][0]
            JTerms["activationDtTerm_sc"] = JTerms["activationDtTerm"] / JAll_opt[0][0]
            JTerms["forceDtTerm_sc"] = JTerms["forceDtTerm"] / JAll_opt[0][0]
        if withArms:
            JTerms["armExcitationTerm_sc"] = JTerms["armExcitationTerm"] / JAll_opt[0][0]
        if withLumbarCoordinateActuators:
            JTerms["lumbarExcitationTerm_sc"] = JTerms["lumbarExcitationTerm"] / JAll_opt[0][0]
        JTerms["jointAccelerationTerm_sc"] = JTerms["jointAccelerationTerm"] / JAll_opt[0][0]
        
        JTerms["positionTerm_sc"] = JTerms["positionTerm"] / JAll_opt[0][0]
        JTerms["velocityTerm_sc"] = JTerms["velocityTerm"] / JAll_opt[0][0]
        if trackQdds:
            JTerms["accelerationTerm_sc"] = JTerms["accelerationTerm"] / JAll_opt[0][0]                
        # Print out contributions to the cost function.
        print("\nContributions to the objective function:")
        if torque_driven_model:
            print("\tCoordinate excitations: {}%".format(np.round(JTerms["coordinateExcitationTerm_sc"] * 100, 2)))
        else:
            print("\tMuscle activations: {}%".format(np.round(JTerms["activationTerm_sc"] * 100, 2)))
            print("\tMuscle activation derivatives: {}%".format(np.round(JTerms["activationDtTerm_sc"] * 100, 2)))
            print("\tMuscle-tendon force derivatives: {}%".format(np.round(JTerms["forceDtTerm_sc"] * 100, 2)))
        if withArms:
            print("\tArm excitations: {}%".format(np.round(JTerms["armExcitationTerm_sc"] * 100, 2)))
        if withLumbarCoordinateActuators:
            print("\tLumbar excitations: {}%".format(np.round(JTerms["lumbarExcitationTerm_sc"] * 100, 2)))
        print("\tJoint accelerations: {}%".format(np.round(JTerms["jointAccelerationTerm_sc"] * 100, 2)))        
        print("\tPosition tracking: {}%".format(np.round(JTerms["positionTerm_sc"] * 100, 2)))
        print("\tVelocity tracking: {}%".format(np.round(JTerms["velocityTerm_sc"] * 100, 2)))
        if trackQdds:
            print("\tAcceleration tracking: {}%".format(np.round(JTerms["accelerationTerm_sc"] * 100, 2)))           
        print("\nNumber of iterations: {}\n".format(stats["iter_count"]))

        #TODO_footVelocity
        if weights['calcnVelocityTerm'] > 0:
            print("\tCalcaneus velocity: {}%".format(np.round((calcnVelTerm_opt_all / JAll_opt[0][0]) * 100, 2)))
            
        # %% Compute knee adduction moments.
        if computeKAM:            
            sys.path.append(os.path.join(baseDir, 'OpenSimPipeline',
                                         'JointReaction'))
            from computeJointLoading import computeKAM
            KAM_labels = ['knee_adduction_r', 'knee_adduction_l']
            IDPath = os.path.join(
                pathResults, 'kinetics_{}_{}.mot'.format(trialName, case))
            IKPath = os.path.join(
                pathResults, 
                'kinematics_activations_{}_{}.mot'.format(trialName, case))
            GRFPath = os.path.join(
                pathResults, 'GRF_{}_{}.mot'.format(trialName, case))
            c_KAM = computeKAM(pathGenericTemplates,
                               pathResults, pathModelFile, IDPath, 
                               IKPath, GRFPath, grfType='sphere',
                               contactSides=contactSides,
                               contactSpheres=contactSpheres,
                               Qds=Qds_opt_nsc.T)
            KAM = np.concatenate(
                (np.expand_dims(c_KAM['KAM_r'], axis=1),
                 np.expand_dims(c_KAM['KAM_l'], axis=1)), axis=1).T              
                
        # %% Compute medial knee contact forces.
        if torque_driven_model:
            computeMCF = False
            print("To compute medial knee contact forces, use a muscle-driven model.\n")
        if computeMCF:
            # Export muscle forces and non muscle-driven torques (if existing).
            labels = ['time'] 
            labels += bothSidesMuscles
            # Muscle forces.
            data = np.concatenate((tgridf.T[:-1], Ft_opt.T),axis=1)
            # Extract non muscle-driven torques (reserve actuators, limit 
            # torques, torques for coordinate actuators, passive torques).
            labels_torques = []
            data_torques = pd.DataFrame()
            if withReserveActuators:
                for count_j, c_j in enumerate(reserveActuatorCoordinates):                    
                    if c_j in data_torques:
                        data_torques[c_j] += rAct_opt_nsc[c_j]
                    else:
                        labels_torques.append(c_j)
                        data_torques.insert(data_torques.shape[1], 
                                            c_j, rAct_opt_nsc[c_j].flatten())
            if enableLimitTorques:
                for count_j, c_j in enumerate(passiveTorqueJoints):
                    if c_j in data_torques:
                        data_torques[c_j] += pT_opt[count_j,:]
                    else:
                        labels_torques.append(c_j)
                        data_torques.insert(data_torques.shape[1], 
                                            c_j, pT_opt[count_j,:])
            if withLumbarCoordinateActuators:
                for count_j, c_j in enumerate(lumbarJoints):
                    aLumbar_opt_nsc = (
                        scaling['LumbarE'].iloc[0][c_j] * 
                        aLumbar_opt[count_j,:-1])
                    if c_j in data_torques:
                        data_torques[c_j] += aLumbar_opt_nsc
                    else:
                        labels_torques.append(c_j)
                        data_torques.insert(data_torques.shape[1],
                                            c_j, aLumbar_opt_nsc)
                    assert np.all(
                            np.abs(torques_opt[joints.index(c_j),:]
                                   - data_torques[c_j]) 
                            < 10**(-2)), "error torques coordinate actuators"
            if withArms:
                for count_j, c_j in enumerate(armJoints):
                    aArm_opt_nsc = (scaling['ArmE'].iloc[0][c_j] * 
                                    aArm_opt[count_j,:-1])
                    c_torque = linearPassiveTorqueArms_opt[count_j,:-1]
                    if c_j in data_torques:
                        data_torques[c_j] += (aArm_opt_nsc + c_torque)
                    else:
                        labels_torques.append(c_j)
                        data_torques.insert(data_torques.shape[1], c_j, 
                                            aArm_opt_nsc + c_torque)
                    assert np.all(
                            np.abs(torques_opt[joints.index(c_j),:] 
                                   - data_torques[c_j]) 
                            < 10**(-2)), "error torques arms"
            # Sanity check for muscle-driven joints
            for count_j, c_j in enumerate(muscleDrivenJoints):
                if c_j in data_torques:
                    c_data_torques = data_torques[c_j].to_numpy()
                else:
                    c_data_torques = np.zeros((data_torques.shape[0],))
                assert np.all(
                        np.abs(torques_opt[joints.index(c_j),:] - (
                            c_data_torques + pMT_opt[count_j, :] + 
                            aMT_opt[count_j, :])) 
                        < 10**(-2)), "error torques muscle-driven joints"
            data_torques_np = data_torques.to_numpy()
            if len(data_torques) > 0:
                data = np.concatenate((data, data_torques_np),axis=1)
                labels += labels_torques
            numpy_to_storage(labels, data, os.path.join(
                pathResults, 'forces_{}_{}.mot'.format(trialName, case)),
                datatype='muscle_forces')
            # Compute medial knee contact forces.
            if not computeKAM:            
                sys.path.append(os.path.join(baseDir, 'OpenSimPipeline',
                                             'JointReaction'))
            from computeJointLoading import computeMCF
            MCF_labels = ['medial_knee_contact_force_r', 
                          'medial_knee_contact_force_l']
            forcePath = os.path.join(pathResults, 
                'forces_{}_{}.mot'.format(trialName, case))
            IK_act_Path = os.path.join(pathResults, 
                'kinematics_activations_{}_{}.mot'.format(trialName, case))
            GRFPath = os.path.join(
                pathResults, 'GRF_{}_{}.mot'.format(trialName, case))                
            c_MCF = computeMCF(pathGenericTemplates, pathResults, 
                               pathModelFile, IK_act_Path, 
                               IK_act_Path, GRFPath, grfType='sphere',
                               contactSides=contactSides,
                               contactSpheres=contactSpheres,
                               muscleForceFilePath=forcePath,
                               pathReserveGeneralizedForces=forcePath,
                               Qds=Qds_opt_nsc.T,
                               replaceMuscles=True)
            MCF = np.concatenate(
                (np.expand_dims(c_MCF['MCF_r'], axis=1),
                 np.expand_dims(c_MCF['MCF_l'], axis=1)), axis=1).T
                
        # %% Express forces in %BW and torques in %BW*height.
        gravity = 9.80665
        BW = settings['mass_kg'] * gravity
        BW_ht = BW * settings['height_m']
        GRF_BW_all_opt = GRF_all_opt['all'] / BW * 100
        GRM_BWht_all_opt = GRM_all_opt['all'] / BW_ht * 100
        torques_BWht_opt = np.zeros(torques_opt.shape)
        for c_j, c_j_name in enumerate(joints):
            if c_j_name in groundPelvisJointsForces:
                torques_BWht_opt[c_j, :] = torques_opt[c_j, :] / BW * 100
            else:
                torques_BWht_opt[c_j, :] = torques_opt[c_j, :] / BW_ht * 100
        if computeKAM:
            KAM_BWht = KAM / BW_ht * 100
        if computeMCF:
            MCF_BW = MCF / BW * 100
        if experimental_force_data_available:
            experimental_grf_BW = experimental_grf / BW * 100
            experimental_grm_BWht = experimental_grm / BW_ht * 100
        if mocap_id_data_available:
            mocap_id_BWht = np.zeros(mocap_id.shape)
            for c_j, c_j_name in enumerate(joints):
                if c_j_name in groundPelvisJointsForces:
                    mocap_id_BWht[c_j, :] = mocap_id[c_j, :] / BW * 100
                else:
                    mocap_id_BWht[c_j, :] = mocap_id[c_j, :] / BW_ht * 100

        # %% Compute joint powers.
        poweredJoints = []
        for joint in joints:
            if not joint in groundPelvisJoints:
                poweredJoints.append(joint)
        idxPoweredJoints = getIndices(joints, poweredJoints)
        # Powers (W) = Torques (Nm) * Angular velocities (rad/s).
        powers_opts = (torques_opt[idxPoweredJoints, :] 
                       * Qds_opt_nsc[idxPoweredJoints, :-1])  
            
        # %% Save optimal trajectories.
        if not os.path.exists(os.path.join(pathResults,
                                           'optimaltrajectories.npy')): 
                optimaltrajectories = {}
        else:  
            optimaltrajectories = np.load(
                    os.path.join(pathResults, 'optimaltrajectories.npy'),
                    allow_pickle=True)   
            optimaltrajectories = optimaltrajectories.item()
        optimaltrajectories[case] = {
            'coordinate_values_toTrack': refData_offset_nsc,
            'coordinate_values': Qs_opt_nsc,
            'coordinate_speeds_toTrack': refData_Qds_nsc,
            'coordinate_speeds': Qds_opt_nsc, 
            'coordinate_accelerations_toTrack': refData_Qdds_nsc,
            'coordinate_accelerations': Qdds_opt_nsc,
            'torques': torques_opt,
            'torques_BWht': torques_BWht_opt,
            'powers': powers_opts,
            'GRF': GRF_all_opt['all'],
            'GRF_BW': GRF_BW_all_opt,
            'GRM': GRM_all_opt['all'],
            'GRM_BWht': GRM_BWht_all_opt,
            'COP': COP_all_opt['all'],
            'freeM': freeT_all_opt['all'],
            'coordinates': joints,
            'coordinates_power': poweredJoints,
            'rotationalCoordinates': rotationalJoints,
            'GRF_labels': GRF_labels_fig,
            'GRM_labels': GRM_labels_fig,
            'COP_labels': COP_labels_fig,
            'time': tgridf,
            'muscles': bothSidesMuscles,
            'passive_limit_torques': pT_opt,
            'muscle_driven_joints': muscleDrivenJoints,
            'limit_torques_joints': passiveTorqueJoints}             
        if computeKAM:
            optimaltrajectories[case]['KAM'] = KAM
            optimaltrajectories[case]['KAM_BWht'] = KAM_BWht
            optimaltrajectories[case]['KAM_labels'] = KAM_labels
        if computeMCF:
            optimaltrajectories[case]['MCF'] = MCF
            optimaltrajectories[case]['MCF_BW'] = MCF_BW
            optimaltrajectories[case]['MCF_labels'] = MCF_labels
        if experimental_force_data_available:
            optimaltrajectories[case]['GRF_experimental'] = experimental_grf
            optimaltrajectories[case]['GRF_BW_experimental'] = experimental_grf_BW
            optimaltrajectories[case]['GRM_experimental'] = experimental_grm
            optimaltrajectories[case]['GRM_BWht_experimental'] = experimental_grm_BWht
        if experimental_emg_data_available:
            optimaltrajectories[case]['muscle_activations_emg'] = experimental_emg
        if mocap_ik_data_available:
            optimaltrajectories[case]['coordinate_values_mocap'] = Qs_mocap_ref
            optimaltrajectories[case]['coordinate_speeds_mocap'] = Qds_mocap_ref
            optimaltrajectories[case]['coordinate_accelerations_mocap'] = Qdds_mocap_ref
        if mocap_id_data_available:
            optimaltrajectories[case]['torques_mocap'] = mocap_id
            optimaltrajectories[case]['torques_BWht_mocap'] = mocap_id_BWht
        optimaltrajectories[case]['iter'] = stats['iter_count']

        if torque_driven_model:
            optimaltrajectories[case]['coordinate_activations'] = aCoord_opt_nsc
        else:
            optimaltrajectories[case]['muscle_activations'] = a_opt
            optimaltrajectories[case]['muscle_forces'] = Ft_opt  
            optimaltrajectories[case]['passive_muscle_torques'] = pMT_opt
            optimaltrajectories[case]['passive_muscle_torques'] = aMT_opt
                
        np.save(os.path.join(pathResults, 'optimaltrajectories.npy'),
                optimaltrajectories)
