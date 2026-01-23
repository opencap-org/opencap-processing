'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsKinematics.py
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
'''

import os
import opensim
import copy
import utils
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate


from utilsProcessing import lowPassFilter
from utilsTRC import trc_2_dict
import numpy as np
from scipy.spatial.transform import Rotation


class kinematics:
    
    def __init__(self, sessionDir, trialName, 
                 modelName=None,
                 lowpass_cutoff_frequency_for_coordinate_values=-1):
        
        self.lowpass_cutoff_frequency_for_coordinate_values = (
            lowpass_cutoff_frequency_for_coordinate_values)
        
        # Model.
        opensim.Logger.setLevelString('error')
        
        modelBasePath = os.path.join(sessionDir, 'OpenSimData', 'Model')
        
        # Check if this is a mono session (models stored in trial subfolders)
        # Check specifically for a subfolder matching the trial name
        isMono = False
        if os.path.exists(modelBasePath):
            trialModelPath = os.path.join(modelBasePath, trialName)
            if os.path.isdir(trialModelPath):
                isMono = True
        
        # Load model if specified, otherwise load the one that was on server
        if modelName is None:
            if isMono:
                # For mono sessions, look in the trial subfolder
                trialModelPath = os.path.join(modelBasePath, trialName)
                if os.path.exists(trialModelPath):
                    # Find .osim file in the trial subfolder
                    osimFiles = [f for f in os.listdir(trialModelPath) if f.endswith('.osim')]
                    if osimFiles:
                        modelPath = os.path.join(trialModelPath, osimFiles[0])
                    else:
                        raise Exception('No .osim file found in ' + trialModelPath)
                else:
                    raise Exception('Trial model folder does not exist: ' + trialModelPath)
            else:
                modelName = utils.get_model_name_from_metadata(sessionDir)
                modelPath = os.path.join(modelBasePath, modelName)
        else:
            if isMono:
                # For mono sessions, look in the trial subfolder
                trialModelPath = os.path.join(modelBasePath, trialName)
                if not modelName.endswith('.osim'):
                    modelName = modelName + '.osim'
                modelPath = os.path.join(trialModelPath, modelName)
            else:
                if not modelName.endswith('.osim'):
                    modelPath = os.path.join(modelBasePath, '{}.osim'.format(modelName))
                else:
                    modelPath = os.path.join(modelBasePath, modelName)
            
        # make sure model exists
        if not os.path.exists(modelPath):
            raise Exception('Model path: ' + modelPath + ' does not exist.')

        self.modelPath = modelPath
        self.model = opensim.Model(modelPath)
        self.model.initSystem()
        
        # Motion file with coordinate values.
        motionPath = os.path.join(sessionDir, 'OpenSimData', 'Kinematics',
                                  '{}.mot'.format(trialName))
        
        # Create time-series table with coordinate values.             
        self.table = opensim.TimeSeriesTable(motionPath)        
        tableProcessor = opensim.TableProcessor(self.table)
        self.columnLabels = list(self.table.getColumnLabels())
        tableProcessor.append(opensim.TabOpUseAbsoluteStateNames())
        self.time = np.asarray(self.table.getIndependentColumn())
        
        # Initialize the state trajectory. We will set it in other functions
        # if it is needed.
        self._stateTrajectory = None
        
        # Filter coordinate values.
        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            tableProcessor.append(
                opensim.TabOpLowPassFilter(
                    lowpass_cutoff_frequency_for_coordinate_values))

        # Convert in radians.
        self.table = tableProcessor.processAndConvertToRadians(self.model)
        
        # Trim if filtered.
        if lowpass_cutoff_frequency_for_coordinate_values > 0:
            time_temp = self.table.getIndependentColumn()            
            self.table.trim(
                time_temp[self.table.getNearestRowIndexForTime(self.time[0])],
                time_temp[self.table.getNearestRowIndexForTime(self.time[-1])])
                
        # Compute coordinate speeds and accelerations and add speeds to table.        
        self.Qs = self.table.getMatrix().to_numpy()
        self.Qds = np.zeros(self.Qs.shape)
        self.Qdds = np.zeros(self.Qs.shape)
        columnAbsoluteLabels = list(self.table.getColumnLabels())
        for i, columnLabel in enumerate(columnAbsoluteLabels):
            spline = interpolate.InterpolatedUnivariateSpline(
                self.time, self.Qs[:,i], k=3)
            # Coordinate speeds
            splineD1 = spline.derivative(n=1)
            self.Qds[:,i] = splineD1(self.time)
            # Coordinate accelerations.
            splineD2 = spline.derivative(n=2)
            self.Qdds[:,i] = splineD2(self.time)            
            # Add coordinate speeds to table.
            columnLabel_speed = columnLabel[:-5] + 'speed'
            self.table.appendColumn(
                columnLabel_speed, 
                opensim.Vector(self.Qds[:,i].flatten().tolist()))
            
        # Append missing muscle states to table.
        # Needed for StatesTrajectory.
        stateVariableNames = self.model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]
        existingLabels = self.table.getColumnLabels()
        for stateVariableNameStr in stateVariableNamesStr:
            if not stateVariableNameStr in existingLabels:
                vec_0 = opensim.Vector([0] * self.table.getNumRows())            
                self.table.appendColumn(stateVariableNameStr, vec_0)
                       
        # Number of muscles.
        self.nMuscles = 0
        self.forceSet = self.model.getForceSet()
        for i in range(self.forceSet.getSize()):        
            c_force_elt = self.forceSet.get(i)  
            if 'Muscle' in c_force_elt.getConcreteClassName():
                self.nMuscles += 1
                
        # Coordinates.
        self.coordinateSet = self.model.getCoordinateSet()
        self.nCoordinates = self.coordinateSet.getSize()
        self.coordinates = [self.coordinateSet.get(i).getName() 
                            for i in range(self.nCoordinates)]
            
        # Find rotational and translational coordinates.
        self.idxColumnTrLabels = [
            self.columnLabels.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 2]
        self.idxColumnRotLabels = [
            self.columnLabels.index(i) for i in self.coordinates if \
            self.coordinateSet.get(i).getMotionType() == 1]
        
        # TODO: hard coded
        self.rootCoordinates = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
            'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        
        self.lumbarCoordinates = ['lumbar_extension', 'lumbar_bending', 
                                  'lumbar_rotation']
        
        self.armCoordinates = ['arm_flex_r', 'arm_add_r', 'arm_rot_r', 
                               'elbow_flex_r', 'pro_sup_r', 
                               'arm_flex_l', 'arm_add_l', 'arm_rot_l', 
                               'elbow_flex_l', 'pro_sup_l']
    
    # Only set the state trajectory when needed because it is slow.
    def stateTrajectory(self):
        if self._stateTrajectory is None:
            self._stateTrajectory = (
                opensim.StatesTrajectory.createFromStatesTable(
                    self.model, self.table))
        return self._stateTrajectory
    
    def get_marker_dict(self, session_dir, trial_name, 
                        lowpass_cutoff_frequency=-1):
        
        trcFilePath = os.path.join(session_dir,
                                   'MarkerData',
                                   '{}.trc'.format(trial_name))
        
        markerDict = trc_2_dict(trcFilePath)
        if lowpass_cutoff_frequency > 0:
            markerDict['markers'] = {
                marker_name: lowPassFilter(self.time, data, lowpass_cutoff_frequency) 
                for marker_name, data in markerDict['markers'].items()}
        
        return markerDict
    
    def rotate_marker_dict(self, markerDict, euler_angles):
        # euler_angles is a dictionary with keys being the axes of rotation
        # (x, y, z) and values being the angles in degrees. e.g. {'x': 90, 'y': 180}
        
        rotated_marker_dict = copy.deepcopy(markerDict)
        rotated_marker_dict['markers'] = {}
        
        rotation = Rotation.from_euler(''.join(list(euler_angles.keys())),
                                       list(euler_angles.values()), degrees=True)
        
        for marker, positions in markerDict['markers'].items():
            rotated_positions = rotation.apply(positions)
            rotated_marker_dict['markers'][marker] = rotated_positions
        
        return rotated_marker_dict
        
    def rotate_com(self, comValues, euler_angles):
        # euler_angles is a dictionary with keys being the axes of rotation
        # (x, y, z) and values being the angles in degrees. e.g. {'x': 90, 'y': 180}
        
        rotation = Rotation.from_euler(''.join(list(euler_angles.keys())),
                                       list(euler_angles.values()), degrees=True)
        
        # turn the x, y, z dataframe entries into a into a 3xN array
        comValuesArray = comValues[['x','y','z']].to_numpy()

        rotated_com = rotation.apply(comValuesArray)

        # turn back into a dataframe with time as first column 
        rotated_com = pd.DataFrame(data=np.concatenate((np.expand_dims(comValues['time'].to_numpy(), axis=1), rotated_com), axis=1),
                                   columns=['time','x','y','z'])
               
        return rotated_com

    def get_coordinate_values(self, in_degrees=True, 
                              lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qs = np.zeros((self.Qs.shape))
            Qs[:, self.idxColumnTrLabels] = self.Qs[:, self.idxColumnTrLabels]
            Qs[:, self.idxColumnRotLabels] = (
                self.Qs[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qs = self.Qs
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qs = lowPassFilter(self.time, Qs, lowpass_cutoff_frequency)
            if self.lowpass_cutoff_frequency_for_coordinate_values > 0:
                print("Warning: You are filtering the coordinate values a second time; coordinate values were filtered when creating your class object.")
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qs), axis=1)
        columns = ['time'] + self.columnLabels            
        self.coordinate_values = pd.DataFrame(data=data, columns=columns)
        
        return self.coordinate_values
    
    def get_coordinate_speeds(self, in_degrees=True, 
                              lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qds = np.zeros((self.Qds.shape))
            Qds[:, self.idxColumnTrLabels] = (
                self.Qds[:, self.idxColumnTrLabels])
            Qds[:, self.idxColumnRotLabels] = (
                self.Qds[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qds = self.Qds
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qds = lowPassFilter(self.time, Qds, lowpass_cutoff_frequency)
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qds), axis=1)
        columns = ['time'] + self.columnLabels            
        coordinate_speeds = pd.DataFrame(data=data, columns=columns)
        
        return coordinate_speeds
    
    def get_coordinate_accelerations(self, in_degrees=True, 
                                     lowpass_cutoff_frequency=-1):
        
        # Convert to degrees.
        if in_degrees:
            Qdds = np.zeros((self.Qdds.shape))
            Qdds[:, self.idxColumnTrLabels] = (
                self.Qdds[:, self.idxColumnTrLabels])
            Qdds[:, self.idxColumnRotLabels] = (
                self.Qdds[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qdds = self.Qdds
            
        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qdds = lowPassFilter(self.time, Qdds, lowpass_cutoff_frequency)
        
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qdds), axis=1)
        columns = ['time'] + self.columnLabels            
        coordinate_accelerations = pd.DataFrame(data=data, columns=columns)
        
        return coordinate_accelerations
    
    def get_muscle_tendon_lengths(self, lowpass_cutoff_frequency=-1):
        
        # Compute muscle-tendon lengths.
        lMT = np.zeros((self.table.getNumRows(), self.nMuscles))
        for i in range(self.table.getNumRows()):
            self.model.realizePosition(self.stateTrajectory()[i])
            if i == 0:
                muscleNames = [] 
            for m in range(self.forceSet.getSize()):        
                c_force_elt = self.forceSet.get(m)  
                if 'Muscle' in c_force_elt.getConcreteClassName():
                    cObj = opensim.Muscle.safeDownCast(c_force_elt)            
                    lMT[i,m] = cObj.getLength(self.stateTrajectory()[i])
                    if i == 0:
                        muscleNames.append(c_force_elt.getName())
                        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            lMT = lowPassFilter(self.time, lMT, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), lMT), axis=1)
        columns = ['time'] + muscleNames               
        muscle_tendon_lengths = pd.DataFrame(data=data, columns=columns)
        
        return muscle_tendon_lengths
    
    def get_moment_arms(self, lowpass_cutoff_frequency=-1):
        
        # Compute moment arms.
        dM =  np.zeros((self.table.getNumRows(), self.nMuscles, 
                        self.nCoordinates))
        for i in range(self.table.getNumRows()):            
            self.model.realizePosition(self.stateTrajectory()[i])
            if i == 0:
                muscleNames = []
            for m in range(self.forceSet.getSize()):        
                c_force_elt = self.forceSet.get(m)  
                if 'Muscle' in c_force_elt.getConcreteClassName():
                    muscleName = c_force_elt.getName()
                    cObj = opensim.Muscle.safeDownCast(c_force_elt)
                    if i == 0:
                        muscleNames.append(c_force_elt.getName())                    
                    for c, coord in enumerate(self.coordinates):
                        # We use prior knowledge to improve computation speed;
                        # We do not want to compute moment arms that are not
                        # relevant, eg for a muscle of the left side with 
                        # respect to a coordinate of the right side.
                        if muscleName[-2:] == '_l' and coord[-2:] == '_r':
                            dM[i, m, c] = 0
                        elif muscleName[-2:] == '_r' and coord[-2:] == '_l':
                            dM[i, m, c] = 0
                        elif (coord in self.rootCoordinates or 
                              coord in self.lumbarCoordinates or 
                              coord in self.armCoordinates):
                            dM[i, m, c] = 0
                        else:
                            coordinate = self.coordinateSet.get(
                                self.coordinates.index(coord))
                            dM[i, m, c] = cObj.computeMomentArm(
                                self.stateTrajectory()[i], coordinate)
                            
        # Clean numerical artefacts (ie, moment arms smaller than 1e-5 m).
        dM[np.abs(dM) < 1e-5] = 0
        
        # Filter.
        if lowpass_cutoff_frequency > 0:            
            for c, coord in enumerate(self.coordinates):
                dM[:, :, c] = lowPassFilter(self.time, dM[:, :, c], 
                                            lowpass_cutoff_frequency)
        
        # Return as DataFrame.
        moment_arms = {}
        for c, coord in enumerate(self.coordinates):
            data = np.concatenate(
                (np.expand_dims(self.time, axis=1), dM[:,:,c]), axis=1)
            columns = ['time'] + muscleNames
            moment_arms[coord] = pd.DataFrame(data=data, columns=columns)
            
        return moment_arms
    
    def compute_center_of_mass(self):        
        
        # Compute center of mass position and velocity.
        self.com_values = np.zeros((self.table.getNumRows(),3))
        self.com_speeds = np.zeros((self.table.getNumRows(),3))        
        for i in range(self.table.getNumRows()):            
            self.model.realizeVelocity(self.stateTrajectory()[i])
            self.com_values[i,:] = self.model.calcMassCenterPosition(
                self.stateTrajectory()[i]).to_numpy()
            self.com_speeds[i,:] = self.model.calcMassCenterVelocity(
                self.stateTrajectory()[i]).to_numpy()
            
    def get_center_of_mass_values(self, lowpass_cutoff_frequency=-1):
        
        self.compute_center_of_mass()        
        com_v = self.com_values
        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            com_v = lowPassFilter(self.time, com_v, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_v), axis=1)
        columns = ['time'] + ['x','y','z']               
        com_values = pd.DataFrame(data=data, columns=columns)
        
        return com_values
    
    def get_center_of_mass_speeds(self, lowpass_cutoff_frequency=-1):
        
        self.compute_center_of_mass()        
        com_s = self.com_speeds
        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            com_s = lowPassFilter(self.time, com_s, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_s), axis=1)
        columns = ['time'] + ['x','y','z']               
        com_speeds = pd.DataFrame(data=data, columns=columns)
        
        return com_speeds
    
    def get_center_of_mass_accelerations(self, lowpass_cutoff_frequency=-1):
        
        self.compute_center_of_mass()        
        com_s = self.com_speeds
        
        # Accelerations are first time derivative of speeds.
        com_a = np.zeros((com_s.shape))
        for i in range(com_s.shape[1]):
            spline = interpolate.InterpolatedUnivariateSpline(
                self.time, com_s[:,i], k=3)
            splineD1 = spline.derivative(n=1)
            com_a[:,i] = splineD1(self.time)        
        
        # Filter.
        if lowpass_cutoff_frequency > 0:
            com_a = lowPassFilter(self.time, com_a, lowpass_cutoff_frequency)                        
              
        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_a), axis=1)
        columns = ['time'] + ['x','y','z']               
        com_accelerations = pd.DataFrame(data=data, columns=columns)
        
        return com_accelerations 

    def get_body_angular_velocity(self, body_names=None, lowpass_cutoff_frequency=-1,
                                  expressed_in='body'):
        
        body_set = self.model.getBodySet()
        if body_names is None:
            body_names = []
            for i in range(body_set.getSize()):
                print(i)
                body = body_set.get(i)
                body_names.append(body.getName())
        
        bodies = [body_set.get(body_name) for body_name in body_names]           
        ground = self.model.getGround()

        angular_velocity = np.ndarray((self.table.getNumRows(),
                              len(body_names)*3)) # time x bodies x dim
                        
        for i_time in range(self.table.getNumRows()): # loop over time
            state = self.stateTrajectory()[i_time]
            self.model.realizeVelocity(state)
            
            
            for i_body,body in enumerate(bodies):
                ang_vel_in_ground = body.getAngularVelocityInGround(state)
                if expressed_in == 'body':
                    angular_velocity[i_time, i_body*3:i_body*3+3] = ground.expressVectorInAnotherFrame(
                                                          state, ang_vel_in_ground, body
                                                          ).to_numpy()
                elif expressed_in == 'ground':
                    angular_velocity[i_time, i_body*3:i_body*3+3] = ang_vel_in_ground.to_numpy()
                else:
                    raise Exception (expressed_in + ' is not a valid frame to express angular' + 
                                     ' velocity.')
                    
        angular_velocity_filtered = lowPassFilter(self.time, angular_velocity, lowpass_cutoff_frequency)
        
        # Put into a dataframe
        data = np.concatenate((np.expand_dims(self.time, axis=1), angular_velocity_filtered), axis=1)
        columns = ['time']
        for i, body_name in enumerate(body_names):
            columns += [f'{body_name}_x', f'{body_name}_y', f'{body_name}_z']
        angular_velocity_df = pd.DataFrame(data=data, columns=columns)
                                                                                                
        return angular_velocity_df

    def get_ranges_of_motion(self, in_degrees=True, lowpass_cutoff_frequency=-1):
        
        self.get_coordinate_values(
            in_degrees=in_degrees, 
            lowpass_cutoff_frequency=lowpass_cutoff_frequency)
        
        # Compute ranges of motion.        
        ROM = {}
        for c, coord in enumerate(self.coordinates):
            ROM[coord] = {}
            ROM[coord]['min'] = self.coordinate_values[coord].min()
            ROM[coord]['max'] = self.coordinate_values[coord].max()
            ROM[coord]['amplitude'] = (
                self.coordinate_values[coord].max() - 
                self.coordinate_values[coord].min())
            
        return ROM 
