"""
    ---------------------------------------------------------------------------
    OpenCap processing: squat_analysis.py
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
"""
 
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt

from utilsKinematics import kinematics
import opensim

class squat_analysis(kinematics):
    
    def __init__(self, session_dir, trial_name, n_repetitions=-1,
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 trimming_start=0, trimming_end=0):
        
        # Inherit init from kinematics class.
        super().__init__(
            session_dir, 
            trial_name, 
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)
        
        # We might want to trim the start/end of the trial to remove bad data.
        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
                        
        # Marker data load and filter.
        self.markerDict = self.get_marker_dict(
            session_dir, 
            trial_name, 
            lowpass_cutoff_frequency = lowpass_cutoff_frequency_for_coordinate_values)

        # Coordinate values.
        self.coordinateValues = self.get_coordinate_values()
        
        # Body transforms
        self.bodyTransformDict = self.get_body_transform_dict()

        # Making sure time vectors of marker and coordinate data are the same.
        if not np.allclose(self.markerDict['time'], self.coordinateValues['time'], atol=.001, rtol=0):
            raise Exception('Time vectors of marker and coordinate data are not the same.')
        
        if not np.allclose(self.bodyTransformDict['time'], self.coordinateValues['time'], atol=.001, rtol=0):
            raise Exception('Time vectors of body transofrms and coordinate data are not the same.')
            
        # Trim marker, body transforms, and coordinate data.
        if self.trimming_start > 0:
            self.idx_trim_start = np.where(np.round(self.markerDict['time'] - self.trimming_start,6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:,:]
            self.bodyTransformDict['time'] = self.bodyTransformDict['time'][self.idx_trim_start:]
            for body in self.bodyTransformDict['body_transforms']:
                self.bodyTransformDict['body_transforms'][body] = \
                    self.bodyTransformDict['body_transforms'][body][self.idx_trim_start:]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]
        
        if self.trimming_end > 0:
            self.idx_trim_end = np.where(np.round(self.markerDict['time'],6) <= np.round(self.markerDict['time'][-1] - self.trimming_end,6))[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end,:]
            self.bodyTransformDict['time'] = self.bodyTransformDict['time'][self.idx_trim_end:]
            for body in self.bodyTransformDict['body_transforms']:
                self.bodyTransformDict['body_transforms'][body] = \
                    self.bodyTransformDict['body_transforms'][body][self.idx_trim_end:]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]
        
        # Segment squat repetitions.
        self.squatEvents = self.segment_squat(n_repetitions=n_repetitions, visualizeSegmentation=True)
        self.nRepetitions = np.shape(self.squatEvents['eventIdxs'])[0]
        
        # Initialize variables to be lazy loaded.
        self._comValues = None
        
        # Time
        self.time = self.coordinateValues['time'].to_numpy()
    
    # Compute COM trajectory.
    def comValues(self):
        if self._comValues is None:
            self._comValues = self.get_center_of_mass_values()
            if self.trimming_start > 0:
                self._comValues = self._comValues.iloc[self.idx_trim_start:]            
            if self.trimming_end > 0:
                self._comValues = self._comValues.iloc[:self.idx_trim_end]
        return self._comValues
    
    def get_squat_events(self):
        
        return self.squatEvents
    
    def compute_scalars(self, scalarNames, return_all=False):
               
        # Verify that scalarNames are methods in squat_analysis.
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if 'compute_' in entry]
        
        if scalarNames is None:
            print('No scalars defined, these methods are available:')
            print(*possibleMethods)
            return
        
        nonexistant_methods = [entry for entry in scalarNames if 'compute_' + entry not in method_names]
        
        if len(nonexistant_methods) > 0:
            raise Exception(str(['compute_' + a for a in nonexistant_methods]) + ' does not exist in squat_analysis class.')
        
        scalarDict = {}
        for scalarName in scalarNames:
            thisFunction = getattr(self, 'compute_' + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]['value'],
                scalarDict[scalarName]['units']) = thisFunction(return_all=return_all)
        
        return scalarDict
        
    
    def segment_squat(self, n_repetitions=-1, peak_proportion_threshold=0.7, 
                      peak_width_rel_height=0.95, peak_distance_sec=0.5,
                      toe_height_threshold=0.05,
                      visualizeSegmentation=False):

        pelvis_ty = self.coordinateValues['pelvis_ty'].to_numpy()  
        dt = np.mean(np.diff(self.time))

        # Identify minimums.
        pelvSignal = np.array(-pelvis_ty - np.min(-pelvis_ty))
        pelvRange = np.abs(np.max(pelvis_ty) - np.min(pelvis_ty))
        peakThreshold = peak_proportion_threshold * pelvRange
        idxMinPelvTy,_ = find_peaks(pelvSignal, prominence=peakThreshold,
                                    distance=peak_distance_sec/dt)
        peakWidths = peak_widths(pelvSignal, idxMinPelvTy, 
                                 rel_height=peak_width_rel_height)
        
        # Store start and end indices.
        startEndIdxs = []
        for start_ips, end_ips in zip(peakWidths[2], peakWidths[3]):
            startEndIdxs.append([start_ips, end_ips])
        startEndIdxs = np.rint(startEndIdxs).astype(int)
            
        # Limit the number of repetitions.
        if n_repetitions != -1:
            startEndIdxs = startEndIdxs[-n_repetitions:]
            
        # Store start and end event times
        eventTimes = self.time[startEndIdxs]            
        
        if visualizeSegmentation:
            plt.figure()     
            plt.plot(-pelvSignal)
            for c_v, val in enumerate(startEndIdxs):
                plt.plot(val, -pelvSignal[val], marker='o', markerfacecolor='k',
                        markeredgecolor='none', linestyle='none',
                        label='Start/End rep')
                if c_v == 0:
                    plt.legend()
            plt.hlines(-peakWidths[1], peakWidths[2], peakWidths[3], color='k',
                       label='peak start/end')
            plt.xlabel('Frames')
            plt.ylabel('Position [m]')
            plt.title('Vertical pelvis position')
            plt.draw()
        
        # Detect squat type (double leg, single leg right, single leg left)
        # Use toe markers
        eventTypes = []
        markersDict = self.markerDict['markers']
        
        for eventIdx in startEndIdxs:
            lToeYMean = np.mean(markersDict['L_toe_study'][eventIdx[0]:eventIdx[1]+1, 1])
            rToeYMean = np.mean(markersDict['r_toe_study'][eventIdx[0]:eventIdx[1]+1, 1])
            
            if lToeYMean - rToeYMean > toe_height_threshold:
                eventTypes.append('single_leg_r')
            elif rToeYMean - lToeYMean > toe_height_threshold:
                eventTypes.append('single_leg_l')
            else:
                eventTypes.append('double_leg')
        
        # Output.
        squatEvents = {
            'eventIdxs': startEndIdxs,
            'eventTimes': eventTimes,
            'eventNames':['repStart','repEnd'],
            'eventTypes': eventTypes}
        
        return squatEvents
    
    def compute_peak_angle(self, coordinate, peak_type="maximum", return_all=False):
        
        # Verify that the coordinate exists.
        if coordinate not in self.coordinateValues.columns:
            raise Exception(coordinate + ' does not exist in coordinate values. Verify the name of the coordinate.')
        
        # Compute max angle for each repetition.
        peak_angles = np.zeros((self.nRepetitions))
        for i in range(self.nRepetitions):            
            rep_range = self.squatEvents['eventIdxs'][i] 
            if peak_type == "maximum":           
                peak_angles[i] = np.max(self.coordinateValues[coordinate].to_numpy()[rep_range[0]:rep_range[1]+1])
            elif peak_type == "minimum":
                peak_angles[i] = np.min(self.coordinateValues[coordinate].to_numpy()[rep_range[0]:rep_range[1]+1])
            else:
                raise Exception('peak_type must be "maximum" or "minimum".')
        
        # Average across all strides.
        peak_angle_mean = np.mean(peak_angles)
        peak_angle_std = np.std(peak_angles)
        
        # Define units.
        units = 'deg'
        
        if return_all:
            return peak_angles, units
        else:
            return peak_angle_mean, peak_angle_std, units
        
    def compute_ratio_peak_angle(self, coordinate_a, coordinate_b, peak_type="maximum", return_all=False):

        peak_angles_a, units_a = self.compute_peak_angle(coordinate_a, peak_type=peak_type, return_all=True)
        peak_angles_b, units_b = self.compute_peak_angle(coordinate_b, peak_type=peak_type, return_all=True)

        # Verify that units are the same.
        if units_a != units_b:
            raise Exception('Units of the two coordinates are not the same.')

        ratio_angles = np.zeros((self.nRepetitions))
        for i in range(self.nRepetitions):
            ratio_angles[i] = peak_angles_a[i] / peak_angles_b[i] * 100

        # Average across all strides.
        ratio_angle_mean = np.mean(ratio_angles)
        ratio_angle_std = np.std(ratio_angles)

        # Define units 
        units = '%'
        
        if return_all:
            return ratio_angles, units
        else:
            return ratio_angle_mean, ratio_angle_std, units
        
    def compute_range_of_motion(self, coordinate, return_all=False):

        # Verify that the coordinate exists.
        if coordinate not in self.coordinateValues.columns:
            raise Exception(coordinate + ' does not exist in coordinate values. Verify the name of the coordinate.')
        
        # Compute max angle for each repetition.
        ranges_of_motion = np.zeros((self.nRepetitions))
        for i in range(self.nRepetitions):            
            rep_range = self.squatEvents['eventIdxs'][i]       
            ranges_of_motion[i] = (np.max(self.coordinateValues[coordinate].to_numpy()[rep_range[0]:rep_range[1]+1]) - 
                                   np.min(self.coordinateValues[coordinate].to_numpy()[rep_range[0]:rep_range[1]+1]))
        
        # Average across all strides.
        range_of_motion_mean = np.mean(ranges_of_motion)
        range_of_motion_std = np.std(ranges_of_motion)
        
        # Define units.
        units = 'deg'
        
        if return_all:
            return ranges_of_motion, units
        else:
            return range_of_motion_mean, range_of_motion_std, units

    def compute_squat_depth(self, return_all=False):
        pelvis_ty = self.coordinateValues['pelvis_ty'].to_numpy()  
        
        squat_depths = np.zeros((self.nRepetitions))
        for i in range(self.nRepetitions):            
            rep_range = self.squatEvents['eventIdxs'][i]
            
            pelvis_ty_range = pelvis_ty[rep_range[0]:rep_range[1]+1]
            squat_depths[i] = np.max(pelvis_ty_range) - np.min(pelvis_ty_range)
        
        # Average across all squats.
        squat_depths_mean = np.mean(squat_depths)
        squat_depths_std = np.std(squat_depths)
        
        # Define units.
        units = 'm'
        
        if return_all:
            return squat_depths, units
        else:
            return squat_depths_mean, squat_depths_std, units

    def compute_trunk_lean_relative_to_pelvis(self, return_all=False):
        torso_transforms = self.bodyTransformDict['body_transforms']['torso']
        pelvis_transforms = self.bodyTransformDict['body_transforms']['pelvis']
        
        max_trunk_leans = np.zeros((self.nRepetitions))
        for i in range(self.nRepetitions):            
            rep_range = self.squatEvents['eventIdxs'][i]
            
            torso_transforms_range = torso_transforms[rep_range[0]:rep_range[1]+1]
            pelvis_transforms_range = pelvis_transforms[rep_range[0]:rep_range[1]+1]
            
            trunk_leans_range = np.zeros(len(torso_transforms_range))
            for j in range(len(torso_transforms_range)):
                y_axis = opensim.Vec3(0, 1, 0)
                torso_y_in_ground = torso_transforms_range[j].xformFrameVecToBase(y_axis).to_numpy()
                
                z_axis = opensim.Vec3(0, 0, 1)
                pelvis_z_in_ground = pelvis_transforms_range[j].xformFrameVecToBase(z_axis).to_numpy()
                
                acos_deg = np.rad2deg(np.arccos(np.dot(torso_y_in_ground, pelvis_z_in_ground)))
                trunk_leans_range[j] = 90 - acos_deg
            
            # using absolute value for now. perhaps keep track of both positive
            # and negative trunk lean angles (to try to detect a bad side?)
            max_trunk_leans[i] = np.max(np.abs(trunk_leans_range))
            units = 'deg'
            
            if return_all:
                return max_trunk_leans, units
            
            else:
                trunk_lean_mean = np.mean(max_trunk_leans)
                trunk_lean_std = np.std(max_trunk_leans)
                return trunk_lean_mean, trunk_lean_std, units
    
    def compute_trunk_lean_relative_to_ground(self, return_all=False):
        torso_transforms = self.bodyTransformDict['body_transforms']['torso']
        pelvis_transforms = self.bodyTransformDict['body_transforms']['pelvis']
        
        max_trunk_leans = np.zeros((self.nRepetitions))
        for i in range(self.nRepetitions):            
            rep_range = self.squatEvents['eventIdxs'][i]
            
            torso_transforms_range = torso_transforms[rep_range[0]:rep_range[1]+1]
            pelvis_transforms_range = pelvis_transforms[rep_range[0]:rep_range[1]+1]
            
            trunk_leans_range = np.zeros(len(torso_transforms_range))
            for j in range(len(torso_transforms_range)):
                y_axis = opensim.Vec3(0, 1, 0)
                torso_y_in_ground = torso_transforms_range[j].xformFrameVecToBase(y_axis).to_numpy()
                
                # find the heading of the pelvis (in the ground x-z plane)
                x_axis = opensim.Vec3(1, 0, 0)
                pelvis_x_in_ground = pelvis_transforms_range[j].xformFrameVecToBase(x_axis).to_numpy()
                pelvis_x_in_ground_xz_plane = pelvis_x_in_ground
                pelvis_x_in_ground_xz_plane[1] = 0
                pelvis_x_in_ground_xz_plane /= np.linalg.norm(pelvis_x_in_ground_xz_plane)
                
                # find the z-axis for comparison to the torso (z-axis in ground
                # that is rotated to the heading of the pelvis)
                rotated_z_axis = np.cross(pelvis_x_in_ground_xz_plane, np.array([0, 1, 0]))
                
                acos_deg = np.rad2deg(np.arccos(np.dot(torso_y_in_ground, rotated_z_axis)))
                trunk_leans_range[j] = 90 - acos_deg
            
            # using absolute value for now. perhaps keep track of both positive
            # and negative trunk lean angles (to try to detect a bad side?)
            max_trunk_leans[i] = np.max(np.abs(trunk_leans_range))
        
        units = 'deg'
            
        if return_all:
            return max_trunk_leans, units
        
        else:
            trunk_lean_mean = np.mean(max_trunk_leans)
            trunk_lean_std = np.std(max_trunk_leans)
            return trunk_lean_mean, trunk_lean_std, units
        
    def compute_trunk_flexion_relative_to_ground(self, return_all=False):
        torso_transforms = self.bodyTransformDict['body_transforms']['torso']
        pelvis_transforms = self.bodyTransformDict['body_transforms']['pelvis']
        
        max_trunk_flexions = np.zeros((self.nRepetitions))
        for i in range(self.nRepetitions):            
            rep_range = self.squatEvents['eventIdxs'][i]
            
            torso_transforms_range = torso_transforms[rep_range[0]:rep_range[1]+1]
            pelvis_transforms_range = pelvis_transforms[rep_range[0]:rep_range[1]+1]
            
            trunk_flexions_range = np.zeros(len(torso_transforms_range))
            for j in range(len(torso_transforms_range)):
                y_axis = opensim.Vec3(0, 1, 0)
                torso_y_in_ground = torso_transforms_range[j].xformFrameVecToBase(y_axis).to_numpy()
                
                # find the heading of the pelvis (in the ground x-z plane)
                x_axis = opensim.Vec3(1, 0, 0)
                pelvis_x_in_ground = pelvis_transforms_range[j].xformFrameVecToBase(x_axis).to_numpy()
                pelvis_x_in_ground_xz_plane = pelvis_x_in_ground
                pelvis_x_in_ground_xz_plane[1] = 0
                pelvis_x_in_ground_xz_plane /= np.linalg.norm(pelvis_x_in_ground_xz_plane)
                
                acos_deg = np.rad2deg(np.arccos(np.dot(torso_y_in_ground, pelvis_x_in_ground_xz_plane)))
                trunk_flexions_range[j] = 90 - acos_deg
            
            # using absolute value for now. perhaps keep track of both positive
            # and negative trunk lean angles (to try to detect a bad side?)
            max_trunk_flexions[i] = np.max(trunk_flexions_range)
        
        units = 'deg'
            
        if return_all:
            return max_trunk_flexions, units
        
        else:
            max_trunk_flexions_mean = np.mean(max_trunk_flexions)
            max_trunk_flexions_std = np.std(max_trunk_flexions)
            return max_trunk_flexions_mean, max_trunk_flexions_std, units
    
    def get_coordinates_segmented(self):
        
        colNames = self.coordinateValues.columns
        data = self.coordinateValues.to_numpy(copy=True)        
        coordValuesSegmented = []
        for eventIdx in self.squatEvents['eventIdxs']:
            coordValuesSegmented.append(pd.DataFrame(data=data[eventIdx[0]:eventIdx[1]], columns=colNames))
        
        return coordValuesSegmented
    
    def get_center_of_mass_values_segmented(self):

        data = np.vstack((self.comValues()['x'],self.comValues()['y'],self.comValues()['z'])).T        
        colNames = ['com_x', 'com_y', 'com_z']        
        comValuesSegmented = []
        for eventIdx in self.squatEvents['eventIdxs']:
            comValuesSegmented.append(pd.DataFrame(data=data[eventIdx[0]:eventIdx[1]], columns=colNames))
        
        return comValuesSegmented
    
    def get_coordinates_segmented_normalized_time(self):
        
        colNames = self.coordinateValues.columns
        data = self.coordinateValues.to_numpy(copy=True)        
        coordValuesSegmentedNorm = []
        for eventIdx in self.squatEvents['eventIdxs']:            
            coordValues = data[eventIdx[0]:eventIdx[1]]            
            coordValuesSegmentedNorm.append(np.stack([np.interp(np.linspace(0,100,101),
                                    np.linspace(0,100,len(coordValues)),coordValues[:,i]) \
                                    for i in range(coordValues.shape[1])],axis=1))
             
        coordValuesTimeNormalized = {}
        # Average.
        coordVals_mean = np.mean(np.array(coordValuesSegmentedNorm),axis=0)
        coordValuesTimeNormalized['mean'] = pd.DataFrame(data=coordVals_mean, columns=colNames)        
        # Standard deviation.
        if self.nRepetitions > 2:
            coordVals_sd = np.std(np.array(coordValuesSegmentedNorm), axis=0)
            coordValuesTimeNormalized['sd'] = pd.DataFrame(data=coordVals_sd, columns=colNames)
        else:
            coordValuesTimeNormalized['sd'] = None        
        # Indiv.
        coordValuesTimeNormalized['indiv'] = [pd.DataFrame(data=d, columns=colNames) for d in coordValuesSegmentedNorm]
        
        return coordValuesTimeNormalized
    
    def get_center_of_mass_segmented_normalized_time(self):
        
        data = np.vstack((self.comValues()['x'],self.comValues()['y'],self.comValues()['z'])).T        
        colNames = ['com_x', 'com_y', 'com_z']        
        comValuesSegmentedNorm = []
        for eventIdx in self.squatEvents['eventIdxs']:            
            comValues = data[eventIdx[0]:eventIdx[1]]            
            comValuesSegmentedNorm.append(np.stack([np.interp(np.linspace(0,100,101),
                                    np.linspace(0,100,len(comValues)),comValues[:,i]) \
                                    for i in range(comValues.shape[1])],axis=1))
             
        comValuesTimeNormalized = {}
        # Average.
        comValues_mean = np.mean(np.array(comValuesSegmentedNorm),axis=0)
        comValuesTimeNormalized['mean'] = pd.DataFrame(data=comValues_mean, columns=colNames)        
        # Standard deviation.
        if self.nRepetitions > 2:
            comValues_sd = np.std(np.array(comValuesSegmentedNorm), axis=0)
            comValuesTimeNormalized['sd'] = pd.DataFrame(data=comValues_sd, columns=colNames)
        else:
            comValuesTimeNormalized['sd'] = None        
        # Indiv.
        comValuesTimeNormalized['indiv'] = [pd.DataFrame(data=d, columns=colNames) for d in comValuesSegmentedNorm]
        
        return comValuesTimeNormalized 
    