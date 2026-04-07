"""
    ---------------------------------------------------------------------------
    OpenCap processing: gait_analysis.py
    ---------------------------------------------------------------------------

    Copyright 2023 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
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
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import copy
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

from utilsKinematics import kinematics


class gait_analysis(kinematics):
    
    def __init__(self, session_dir, trial_name, leg='auto',
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_gait_cycles=-1, gait_style='auto', trimming_start=0, 
                 trimming_end=0):
        
        # Inherit init from kinematics class.
        super().__init__(
            session_dir, 
            trial_name, 
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)
        
        # We might want to trim the start/end of the trial to remove bad data. 
        # For example, this might be needed with HRNet during overground 
        # walking, where, at the end, the subject is leaving the field of view 
        # but HRNet returns relatively high confidence values. As a result,
        # the trial is not well trimmed. Here, we provide the option to
        # manually trim the start and end of the trial.
        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
                        
        # Marker data load and filter.
        self.markerDict = self.get_marker_dict(session_dir, trial_name, 
            lowpass_cutoff_frequency = lowpass_cutoff_frequency_for_coordinate_values)

        # Coordinate values.
        self.coordinateValues = self.get_coordinate_values()
        
        # Trim marker data and coordinate values.
        if self.trimming_start > 0:
            self.idx_trim_start = np.where(np.round(self.markerDict['time'] - self.trimming_start,6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:,]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:,:]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]
        
        if self.trimming_end > 0:
            self.idx_trim_end = np.where(np.round(self.markerDict['time'],6) <= np.round(self.markerDict['time'][-1] - self.trimming_end,6))[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end,]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end,:]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]
        
        # Rotate marker data so x is forward (not using for now, but could be useful for some analyses).
        self.rotation_about_y, self.markerDictRotated = self.rotate_x_forward()

        # Segment gait cycles.
        self.gaitEvents = self.segment_walking(n_gait_cycles=n_gait_cycles,leg=leg)
        self.nGaitCycles = np.shape(self.gaitEvents['ipsilateralIdx'])[0]
        
        # Determine treadmill speed (0 if overground).
        self.treadmillSpeed,_ = self.compute_treadmill_speed(gait_style=gait_style)
        
        # Initialize variables to be lazy loaded.
        self._comValues = None
        self._R_world_to_gait = None
        self._leg_length = None

        # Rotate marker data with a per gait cycle rotation
        self.markerDictRotatedPerGaitCycle = self.rotate_vector_into_gait_frame()
    
    # Compute COM trajectory.
    def comValues(self,rotate=None,filt_freq=-1):
        if rotate == None:
            if self._comValues is None or filt_freq != -1:
                self._comValues = self.get_center_of_mass_values(lowpass_cutoff_frequency = filt_freq)
                if self.trimming_start > 0:
                    self._comValues = self._comValues.iloc[self.idx_trim_start:]            
                if self.trimming_end > 0:
                    self._comValues = self._comValues.iloc[:self.idx_trim_end]
            return self._comValues

        if rotate == 'gaitCycle':
            if self._comValuesRotatedPerGaitCycle is None or filt_freq!=-1:
                comUnrotated = self.comValues(filt_freq=filt_freq)
                comRotated = self.rotate_vector_into_gait_frame(comUnrotated[['x', 'y', 'z']].to_numpy())
                # turn back into a dataframe with time as first column
                self._comValuesRotatedPerGaitCycle = pd.DataFrame(data=np.concatenate((np.expand_dims(comUnrotated['time'].to_numpy(), axis=1), comRotated),axis=1),
                                           columns=['time','x','y','z'])        
                if self.trimming_start > 0:
                    self._comValuesRotatedPerGaitCycle = self._comValuesRotatedPerGaitCycle.iloc[self.idx_trim_start:]            
                if self.trimming_end > 0:
                    self._comValuesRotatedPerGaitCycle = self._comValuesRotatedPerGaitCycle.iloc[:self.idx_trim_end]   
            return self._comValuesRotatedPerGaitCycle
        
        if rotate == 'y': # need to initialize self.rotation_about_y -- currently commented in the init function
            if self._comValuesRotated is None or filt_freq!=-1:
                self._comValuesRotated = self.rotate_com(self.comValues(filt_freq=filt_freq),{'y':self.rotation_about_y})
                if self.trimming_start > 0:
                    self._comValuesRotated = self._comValuesRotated.iloc[self.idx_trim_start:]            
                if self.trimming_end > 0:
                    self._comValuesRotated = self._comValuesRotated.iloc[:self.idx_trim_end]   
            return self._comValuesRotated
    
    # Compute gait frame.
    def R_world_to_gait(self):
        if self._R_world_to_gait is None:
            self._R_world_to_gait = self.compute_gait_frame()
        return self._R_world_to_gait
    
    def get_gait_events(self):
        
        return self.gaitEvents
    
    def rotate_x_forward(self):
        # Find the midpoint of the PSIS markers
        psis_midpoint = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study']) / 2

        # Find the midpoint of the ASIS markers
        asis_midpoint = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study']) / 2

        # Compute the vector pointing from the PSIS midpoint to the ASIS midpoint
        heading_vector = asis_midpoint - psis_midpoint

        # Compute the angle between the heading vector projected onto x-z plane and x-axis
        angle = np.unwrap(np.arctan2(heading_vector[:,2], heading_vector[:,0]))

        # compute average angle during middle 50% of the trial
        n_frames = len(self.markerDict['time'])
        start_index = int(n_frames * 0.25)
        end_index = int(n_frames * 0.75)
        angle = np.degrees(np.mean(angle[start_index:end_index], axis=0))

        # Apply the rotation to the marker data
        marker_dict_rotated = self.rotate_marker_dict(self.markerDict, {'y':angle})

        return angle, marker_dict_rotated
    
        
    def leg_length(self):

        if self._leg_length is None:

            leg, contLeg = self.get_leg()
            # compute the midpoint between the knee and m_knee markers
            kjc = (self.markerDict['markers'][leg + '_knee_study'] + 
                            self.markerDict['markers'][leg + '_mknee_study']) / 2
            ajc = (self.markerDict['markers'][leg + '_ankle_study'] +
                                self.markerDict['markers'][leg + '_mankle_study']) / 2
            hjc = self.markerDict['markers'][leg.upper() + 'HJC_study']
                
            # compute the femur vector from hjc to kjc, then find the average of its norm
            femur_vector = kjc - hjc
            femur_length = np.mean(np.linalg.norm(femur_vector, axis=1))

            # compute the tibia vector from kjc to ajc, then find the average of its norm
            tibia_vector = ajc - kjc
            tibia_length = np.mean(np.linalg.norm(tibia_vector, axis=1))

            # sum the femur and tibia lengths to get the leg length
            _leg_length = {'ipsilateral':femur_length + tibia_length}

            # repeat for contraolateral leg
            kjc = (self.markerDict['markers'][contLeg + '_knee_study'] + 
                            self.markerDict['markers'][contLeg + '_mknee_study']) / 2
            ajc = (self.markerDict['markers'][contLeg + '_ankle_study'] +
                                self.markerDict['markers'][contLeg + '_mankle_study']) / 2
            hjc = self.markerDict['markers'][contLeg.upper() + 'HJC_study']
                   
            femur_vector = kjc - hjc
            femur_length = np.mean(np.linalg.norm(femur_vector, axis=1))

            tibia_vector = ajc - kjc
            tibia_length = np.mean(np.linalg.norm(tibia_vector, axis=1))

            _leg_length['contralateral'] = femur_length + tibia_length
        
        return _leg_length
    
    
    def compute_scalars(self,scalarNames,return_all=False):
               
        # Verify that scalarNames are methods in gait_analysis.
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if 'compute_' in entry]
        
        if scalarNames is None:
            print('No scalars defined, these methods are available:')
            print(*possibleMethods)
            return
        
        nonexistant_methods = [entry for entry in scalarNames if 'compute_' + entry not in method_names]
        
        if len(nonexistant_methods) > 0:
            raise Exception(str(['compute_' + a for a in nonexistant_methods]) + ' does not exist in gait_analysis class.')
        
        scalarDict = {}
        for scalarName in scalarNames:
            thisFunction = getattr(self, 'compute_' + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]['value'],
                scalarDict[scalarName]['units']) = thisFunction(return_all=return_all)
        
        return scalarDict
    
    
    def compute_stride_length(self,return_all=False):
        
        leg,_ = self.get_leg()
        
        calc_position = self.markerDictRotatedPerGaitCycle['markers'][leg + '_calc_study']

        # On treadmill, the stride length is the difference in ipsilateral
        # calcaneus position at heel strike + treadmill speed * time.
        strideLengths = (
                - calc_position[self.gaitEvents['ipsilateralIdx'][:,:1],0] +
                calc_position[self.gaitEvents['ipsilateralIdx'][:,2:3],0] + 
                self.treadmillSpeed * np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)]))       
        
        # Average across all strides.
        strideLength = np.mean(strideLengths)
        
        # Define units.
        units = 'm'
        
        if return_all:
            return strideLengths,units
        else: 
            return strideLength, units
        
    
    def compute_step_length(self,return_all=False):
        leg, contLeg = self.get_leg()
        step_lengths = {}
        
        step_lengths[contLeg.lower()] = (
            - self.markerDictRotated['markers'][leg + '_calc_study'][self.gaitEvents['ipsilateralIdx'][:,:1],0] + 
            self.markerDictRotated['markers'][contLeg + '_calc_study'][self.gaitEvents['contralateralIdx'][:,1:2],0] + 
            self.treadmillSpeed * (self.gaitEvents['contralateralTime'][:,1:2] -
                                   self.gaitEvents['ipsilateralTime'][:,:1]))
        
        step_lengths[leg.lower()]  = (
            self.markerDictRotated['markers'][leg + '_calc_study'][self.gaitEvents['ipsilateralIdx'][:,2:],0] - 
            self.markerDictRotated['markers'][contLeg + '_calc_study'][self.gaitEvents['contralateralIdx'][:,1:2],0] + 
            self.treadmillSpeed * (-self.gaitEvents['contralateralTime'][:,1:2] +
                                   self.gaitEvents['ipsilateralTime'][:,2:]))
               
        # Average across all strides.
        step_length = {key: np.mean(values) for key, values in step_lengths.items()}
        
        # Define units.
        units = 'm'
        
        # some functions depend on having values for each step, otherwise return average
        if return_all:
            return step_lengths, units
        else:
            return step_length, units
        
        
    def compute_step_length_symmetry(self,return_all=False):
        step_lengths,units = self.compute_step_length(return_all=True)
        
        step_length_symmetry_all = step_lengths['r'] / step_lengths['l'] * 100
        
        # Average across strides
        step_length_symmetry = np.mean(step_length_symmetry_all)
        
        # define units 
        units = '% (R/L)'
        
        if return_all:
            return step_length_symmetry_all, units
        else:
            return step_length_symmetry, units
        
    
    def compute_gait_speed(self,return_all=False):
                           
        comValuesArray = np.vstack((self.comValues()['x'],self.comValues()['y'],self.comValues()['z'])).T
        gait_speeds = (
            np.linalg.norm(
                comValuesArray[self.gaitEvents['ipsilateralIdx'][:,:1]] -
                comValuesArray[self.gaitEvents['ipsilateralIdx'][:,2:3]], axis=2) /
                np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)]) + self.treadmillSpeed) 
        
        # Average across all strides.
        gait_speed = np.mean(gait_speeds)
        
        # Define units.
        units = 'm/s'
        
        if return_all:
            return gait_speeds,units
        else:
            return gait_speed, units
    
    def compute_cadence(self,return_all=False):
        
        # In steps per minute.
        cadence_all = 60*2/np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)])
        
        # Average across all strides.
        cadence = np.mean(cadence_all)
        
        # Define units.
        units = 'steps/min'
        
        if return_all:
            return cadence_all,units
        else:
            return cadence, units
        
    def compute_treadmill_speed(self, overground_speed_threshold=0.3,
                                gait_style='auto', return_all=False):
        
        # Heuristic to determine if overground or treadmill.
        if gait_style == 'auto' or gait_style == 'treadmill':
            leg,_ = self.get_leg()
            
            foot_position = self.markerDict['markers'][leg + '_ankle_study']
            
            stanceTimeLength = np.round(np.diff(self.gaitEvents['ipsilateralIdx'][:,:2]))
            startIdx = np.round(self.gaitEvents['ipsilateralIdx'][:,:1]+.1*stanceTimeLength).astype(int)
            endIdx = np.round(self.gaitEvents['ipsilateralIdx'][:,1:2]-.3*stanceTimeLength).astype(int)
                
            # Average instantaneous velocities.
            dt = np.diff(self.markerDict['time'][:2])[0]
            treadmillSpeeds = np.zeros((self.nGaitCycles,))
            for i in range(self.nGaitCycles):
                treadmillSpeeds[i,] = np.linalg.norm(np.mean(np.diff(
                    foot_position[startIdx[i,0]:endIdx[i,0],:],axis=0),axis=0)/dt)
            
            treadmillSpeed = np.mean(treadmillSpeeds)
            
            # Overground if treadmill speed is below threshold and gait style not set to treadmill.
            if treadmillSpeed < overground_speed_threshold and not gait_style == 'treadmill':
                treadmillSpeed = 0
                treadmillSpeeds = np.zeros(self.nGaitCycles)
        
        # Overground if gait style set to overground.
        elif gait_style == 'overground':
            treadmillSpeed = 0
            treadmillSpeeds = np.zeros(self.nGaitCycles)
            
        # Define units.
        units = 'm/s'
                           
        if return_all:
            return treadmillSpeeds,units
        else:
            return treadmillSpeed, units
    
    def compute_step_width(self,return_all=False):
        
        leg,contLeg = self.get_leg()
        
        # Get ankle joint center positions.
        ankle_position_ips = (
            self.markerDict['markers'][leg + '_ankle_study'] + 
            self.markerDict['markers'][leg + '_mankle_study'])/2
        ankle_position_cont = (
            self.markerDict['markers'][contLeg + '_ankle_study'] + 
            self.markerDict['markers'][contLeg + '_mankle_study'])/2        
        
        # Find indices of 40-60% of the stance phase
        ips_stance_length = np.diff(self.gaitEvents['ipsilateralIdx'][:,(0,1)])
        cont_stance_length = (self.gaitEvents['contralateralIdx'][:,0] - 
                              self.gaitEvents['ipsilateralIdx'][:,0] +
                              self.gaitEvents['ipsilateralIdx'][:,2]-
                              self.gaitEvents['contralateralIdx'][:,1])
        
        midstanceIdx_ips = [range(self.gaitEvents['ipsilateralIdx'][i,0] + 
                                  int(np.round(.4*ips_stance_length[i])),
                                  self.gaitEvents['ipsilateralIdx'][i,0] + 
                                  int(np.round(.6*ips_stance_length[i]))) 
                                  for i in range(self.nGaitCycles)]
        
        midstanceIdx_cont = [range(np.min((self.gaitEvents['contralateralIdx'][i,1] + 
                                  int(np.round(.4*cont_stance_length[i])),
                                  self.gaitEvents['ipsilateralIdx'][i,2]-1)),
                                  np.min((self.gaitEvents['contralateralIdx'][i,1] + 
                                  int(np.round(.6*cont_stance_length[i])),
                                  self.gaitEvents['ipsilateralIdx'][i,2]))) 
                                  for i in range(self.nGaitCycles)]   
        
        ankleVector = np.zeros((self.nGaitCycles,3))
        for i in range(self.nGaitCycles):
            ankleVector[i,:] = (
                np.mean(ankle_position_cont[midstanceIdx_cont[i],:],axis=0) - 
                np.mean(ankle_position_ips[midstanceIdx_ips[i],:],axis=0))
                     
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector[i,:], self.R_world_to_gait()[i,:,:]) 
            for i in range(self.nGaitCycles)])
        
        # Step width is z distance.
        stepWidths = np.abs(ankleVector_inGaitFrame[:,2])
        
        # Average across all strides.
        stepWidth = np.mean(stepWidths)
        
        # Define units.
        units = 'm'
        
        if return_all:
            return stepWidths, units
        else:
            return stepWidth, units
    
    def compute_stance_time(self, return_all=False):
        
        stanceTimes = np.diff(self.gaitEvents['ipsilateralTime'][:,:2])
        
        # Average across all strides.
        stanceTime = np.mean(stanceTimes)
        
        # Define units.
        units = 's'
        
        if return_all:
            return stanceTimes, units
        else:
            return stanceTime, units
    
    def compute_swing_time(self, return_all=False):
        
        swingTimes = np.diff(self.gaitEvents['ipsilateralTime'][:,1:])
        
        # Average across all strides.
        swingTime = np.mean(swingTimes)
        
        # Define units.
        units = 's'
        
        if return_all:
            return swingTimes, units
        else:  
            return swingTime, units
    
    def compute_single_support_time(self,return_all=False):
        
        double_support_time,_ = self.compute_double_support_time(return_all=True) 

        singleSupportTimes = 100 - double_support_time    
        
        # Average across all strides.
        singleSupportTime = np.mean(singleSupportTimes)
        
        # Define units.
        units = '%'
        
        if return_all:
            return singleSupportTimes,units
        else:
            return singleSupportTime, units
        
    def compute_double_support_time(self,return_all=False):
        
        # Ipsilateral stance time - contralateral swing time.
        doubleSupportTimes = (
            (np.diff(self.gaitEvents['ipsilateralTime'][:,:2]) - 
            np.diff(self.gaitEvents['contralateralTime'][:,:2])) /
            np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)])) * 100
                            
        # Average across all strides.
        doubleSupportTime = np.mean(doubleSupportTimes)
        
        # Define units.
        units = '%'
        
        if return_all:
            return doubleSupportTimes, units
        else:
            return doubleSupportTime, units
        
    def compute_midswing_dorsiflexion_angle(self,return_all=False):
        # compute ankle dorsiflexion angle during midstance
        to_1_idx = self.gaitEvents['ipsilateralIdx'][:,1]
        hs_2_idx = self.gaitEvents['ipsilateralIdx'][:,2]
        
        # ankle markers
        leg,contLeg = self.get_leg()
        ankleVector = (self.markerDict['markers'][leg + '_ankle_study'] - 
                       self.markerDict['markers'][contLeg + '_ankle_study'])
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector, self.R_world_to_gait()[i,:,:]) 
              for i in range(self.nGaitCycles)])                                          
        
        swingDfAngles = np.zeros((to_1_idx.shape))
        
        for i in range(self.nGaitCycles):
            # find index within a swing phase with the smallest z distance between ankles
            idx_midSwing = np.argmin(np.abs(ankleVector_inGaitFrame[
                                     i,to_1_idx[i]:hs_2_idx[i],0]))+to_1_idx[i]
            
            swingDfAngles[i] = np.mean(self.coordinateValues['ankle_angle_' + 
                                self.gaitEvents['ipsilateralLeg']].to_numpy()[idx_midSwing])          
        
        # Average across all strides.
        swingDfAngle = np.mean(swingDfAngles)
        
        # Define units.
        units = 'deg'
        
        if return_all:
            return swingDfAngles, units
        else:
            return swingDfAngle, units
        
    def compute_midswing_ankle_heigh_dif(self,return_all=False):
        # compute vertical clearance of the swing ankle above the stance ankle
        # at the time when the ankles pass by one another
        to_1_idx = self.gaitEvents['ipsilateralIdx'][:,1]
        hs_2_idx = self.gaitEvents['ipsilateralIdx'][:,2]
        
        # ankle markers
        leg,contLeg = self.get_leg()
        ankleVector = (self.markerDict['markers'][leg + '_ankle_study'] - 
                       self.markerDict['markers'][contLeg + '_ankle_study'])
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector, self.R_world_to_gait()[i,:,:]) 
              for i in range(self.nGaitCycles)])                                          
        
        swingAnkleHeighDiffs = np.zeros((to_1_idx.shape))
        
        for i in range(self.nGaitCycles):
            # find index within a swing phase with the smallest z distance between ankles
            idx_midSwing = np.argmin(np.abs(ankleVector_inGaitFrame[
                                     i,to_1_idx[i]:hs_2_idx[i],0]))+to_1_idx[i]
            
            swingAnkleHeighDiffs[i] = ankleVector_inGaitFrame[i,idx_midSwing,1]  
        
        # Average across all strides.
        swingAnkleHeighDiff = np.mean(swingAnkleHeighDiffs)
        
        # Define units.
        units = 'm'
        
        if return_all:
            return swingAnkleHeighDiffs, units
        else:
            return swingAnkleHeighDiff, units
        
    def compute_peak_angle(self,dof,start_idx,end_idx,return_all=False):
        # start_idx and end_idx are 1xnGaitCycles        
        
        peakAngles = np.zeros((self.nGaitCycles))
        
        for i in range(self.nGaitCycles):                       
            peakAngles[i] = np.max(self.coordinateValues[dof + '_' +
                                self.gaitEvents['ipsilateralLeg']][start_idx[i]:end_idx[i]])
        
        # Average across all strides.
        peakAngle = np.mean(peakAngles)
        
        # Define units.
        units = 'deg'
        
        if return_all:
            return peakAngles, units
        else:
            return peakAngle, units
        
    def compute_rom(self,dof,start_idx,end_idx,return_all=False):
        # start_idx and end_idx are 1xnGaitCycles        
        
        roms = np.zeros((self.nGaitCycles))
        
        for i in range(self.nGaitCycles):                       
            roms[i] = np.ptp(self.coordinateValues[dof + '_' +
                                self.gaitEvents['ipsilateralLeg']][start_idx[i]:end_idx[i]])
        
        # Average across all strides.
        rom = np.mean(roms)
        
        # Define units.
        units = 'deg'
        
        if return_all:
            return roms, units
        else:
            return rom, units
                        
    def compute_correlations(self, cols_to_compare=None, visualize=False,
                             return_all=False):
        # this computes a weighted correlation between either side's dofs. 
        # the weighting is based on mean absolute percent error. In effect,
        # this penalizes both shape and magnitude differences.
        
        leg,contLeg = self.get_leg(lower=True)
               
        correlations_all_cycles = []
        mean_correlation_all_cycles = np.zeros((self.nGaitCycles,1))
        
        for i in range(self.nGaitCycles):

            
            hs_ind_1 = self.gaitEvents['ipsilateralIdx'][i,0]
            hs_ind_cont = self.gaitEvents['contralateralIdx'][i,1]
            hs_ind_2 = self.gaitEvents['ipsilateralIdx'][i,2]
            
            df1 = pd.DataFrame()
            df2 = pd.DataFrame()
            
            if cols_to_compare is None:
                cols_to_compare = df1.columns
            
            # create a dataframe of coords for this gait cycle
            for col in self.coordinateValues.columns:
                if col.endswith('_' + leg):
                    df1[col] = self.coordinateValues[col][hs_ind_1:hs_ind_2]
                elif col.endswith('_' + contLeg):
                    df2[col] = np.concatenate((self.coordinateValues[col][hs_ind_cont:hs_ind_2],
                                               self.coordinateValues[col][hs_ind_1:hs_ind_cont]))
            df1 = df1.reset_index(drop=True)
            df2 = df2.reset_index(drop=True)
                    
            # Interpolating both dataframes to have 101 rows for each column
            df1_interpolated = df1.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=100)
            df2_interpolated = df2.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=100)
        
            # Computing the correlation between appropriate columns in both dataframes
            correlations = {}
            total_weighted_correlation = 0
            # total_weight = 0
        
            for col1 in df1_interpolated.columns:
                if any(col1.startswith(col_compare) for col_compare in cols_to_compare):
                    if col1.endswith('_r'):   
                        corresponding_col = col1[:-2] + '_l'
                    elif col1.endswith('_l'):
                        corresponding_col = col1[:-2] + '_r'
                            
                    if corresponding_col in df2_interpolated.columns:
                        signal1 = df1_interpolated[col1]
                        signal2 = df2_interpolated[corresponding_col]
        
                        max_range_signal1 = np.ptp(signal1)
                        max_range_signal2 = np.ptp(signal2)
                        max_range = max(max_range_signal1, max_range_signal2)
        
                        mean_abs_error = np.mean(np.abs(signal1 - signal2)) / max_range
        
                        correlation = signal1.corr(signal2)
                        weight = 1 - mean_abs_error
        
                        weighted_correlation = correlation * weight
                        correlations[col1] = weighted_correlation
        
                        total_weighted_correlation += weighted_correlation
        
                        # Plotting the signals if visualize is True
                        if visualize:
                            plt.figure(figsize=(8, 5))
                            plt.plot(signal1, label='df1')
                            plt.plot(signal2, label='df2')
                            plt.title(f"Comparison between {col1} and {corresponding_col} with weighted correlation {weighted_correlation}")
                            plt.legend()
                            plt.show()
        
            mean_correlation_all_cycles[i] = total_weighted_correlation / len(correlations)
            correlations_all_cycles.append(correlations)
            
        if not return_all:
            mean_correlation_all_cycles = np.mean(mean_correlation_all_cycles)
            correlations_all_cycles =  {key: sum(d[key] for d in correlations_all_cycles) / 
                                        len(correlations_all_cycles) for key in correlations_all_cycles[0]}
            
        return correlations_all_cycles, mean_correlation_all_cycles

    def compute_gait_frame(self):

        # Create frame for each gait cycle with x: pelvis heading, 
        # z: average vector between ASIS during gait cycle, y: cross.
        
        # Pelvis center trajectory (for overground heading vector).
        pelvisMarkerNames = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study']
        pelvisMarkers = [self.markerDict['markers'][mkr]  for mkr in pelvisMarkerNames]
        pelvisCenter = np.mean(np.array(pelvisMarkers),axis=0)
        
        # Ankle trajectory (for treadmill heading vector).
        leg = self.gaitEvents['ipsilateralLeg']
        if leg == 'l': leg='L'
        anklePos = self.markerDict['markers'][leg + '_ankle_study']
        
        # Vector from left ASIS to right ASIS (for mediolateral direction).
        asisMarkerNames = ['L.ASIS_study','r.ASIS_study']
        asisMarkers = [self.markerDict['markers'][mkr]  for mkr in asisMarkerNames]
        asisVector = np.squeeze(np.diff(np.array(asisMarkers),axis=0))
        
        # Heading vector per gait cycle.
        # If overground, use pelvis center trajectory; treadmill: ankle trajectory.
        if self.treadmillSpeed == 0:
            x = np.diff(pelvisCenter[self.gaitEvents['ipsilateralIdx'][:,(0,2)],:],axis=1)[:,0,:]
            x = x / np.linalg.norm(x,axis=1,keepdims=True)
        else: 
            x = np.zeros((self.nGaitCycles,3))
            for i in range(self.nGaitCycles):
                x[i,:] = anklePos[self.gaitEvents['ipsilateralIdx'][i,2]] - \
                         anklePos[self.gaitEvents['ipsilateralIdx'][i,1]]
            x = x / np.linalg.norm(x,axis=1,keepdims=True)
            
        # Mean ASIS vector over gait cycle.
        z_temp = np.zeros((self.nGaitCycles,3))
        for i in range(self.nGaitCycles):
            z_temp[i,:] = np.mean(asisVector[self.gaitEvents['ipsilateralIdx'][i,0]: \
                             self.gaitEvents['ipsilateralIdx'][i,2]],axis=0)
        z_temp = z_temp / np.linalg.norm(z_temp,axis=1,keepdims=True)
        
        # Cross to get y.
        y = np.cross(z_temp,x)
        
        z = np.cross(x,y)
        
        # 3x3xnSteps.
        R_lab_to_gait = np.stack((x.T,y.T,z.T),axis=1).transpose((2, 0, 1))
        
        return R_lab_to_gait
    
    def rotate_vector_into_gait_frame(self,vectorArray=None):
        # vectorArray is a nFramesx3 array
        # This takes a vector array and rotates it into the gait frame, per gait frame. Thus,
        # the data in the vector array is not expressed all in the same frame. This data should
        # only be used on gait cycle, by gait cycle data. Note, the second heel strike data gets overwritten
        # by subsequent gait cycles (since it is the same index as the first heel strike in the subsequent
        # gait cycle). We assume that the gait frame doesn't change dramatically from step to step.

        def rotate_vec(vec,R):
            return np.dot(vec,R)
        
        if vectorArray is None: # rotate each marker in the entire markerDict
            markerDict_rotated_per_step = copy.deepcopy(self.markerDict)
            for marker_name,marker in markerDict_rotated_per_step['markers'].items():
                for i in range(self.nGaitCycles):
                    markerDict_rotated_per_step['markers'][marker_name][self.gaitEvents['ipsilateralIdx'][i,0]:
                                                                        self.gaitEvents['ipsilateralIdx'][i,2],:] = rotate_vec(
                    marker[self.gaitEvents['ipsilateralIdx'][i,0]:self.gaitEvents['ipsilateralIdx'][i,2],:],
                    self.R_world_to_gait()[i,:,:])
            return markerDict_rotated_per_step
            
        else:
            for i in range(self.nGaitCycles):
                vectorArray[self.gaitEvents['ipsilateralIdx'][i,0]:self.gaitEvents['ipsilateralIdx'][i,2],:] = rotate_vec(
                    vectorArray[self.gaitEvents['ipsilateralIdx'][i,0]:self.gaitEvents['ipsilateralIdx'][i,2],:],
                        self.R_world_to_gait()[i,:,:])

            return vectorArray
    
    def get_leg(self,lower=False):

        if self.gaitEvents['ipsilateralLeg'] == 'r':
            leg = 'r'
            contLeg = 'L'
        else:
            leg = 'L'
            contLeg = 'r'
        
        if lower:
            return leg.lower(), contLeg.lower()
        else:
            return leg, contLeg
    
    def get_coordinates_normalized_time(self):
        
        colNames = self.coordinateValues.columns
        data = self.coordinateValues.to_numpy(copy=True)
        coordValuesNorm = []
        for i in range(self.nGaitCycles):
            coordValues = data[self.gaitEvents['ipsilateralIdx'][i,0]:self.gaitEvents['ipsilateralIdx'][i,2]+1]
            coordValuesNorm.append(np.stack([np.interp(np.linspace(0,100,101),
                                   np.linspace(0,100,len(coordValues)),coordValues[:,i]) \
                                   for i in range(coordValues.shape[1])],axis=1))
             
        coordinateValuesTimeNormalized = {}
        # Average.
        coordVals_mean = np.mean(np.array(coordValuesNorm),axis=0)
        coordinateValuesTimeNormalized['mean'] = pd.DataFrame(data=coordVals_mean, columns=colNames)
        
        # Standard deviation.
        if self.nGaitCycles >2:
            coordVals_sd = np.std(np.array(coordValuesNorm), axis=0)
            coordinateValuesTimeNormalized['sd'] = pd.DataFrame(data=coordVals_sd, columns=colNames)
        else:
            coordinateValuesTimeNormalized['sd'] = None
        
        # Return to dataframe.
        coordinateValuesTimeNormalized['indiv'] = [pd.DataFrame(data=d, columns=colNames) for d in coordValuesNorm]
        
        return coordinateValuesTimeNormalized

    def segment_walking(self, n_gait_cycles=-1, leg='auto', visualize=False):

        # n_gait_cycles = -1 finds all accessible gait cycles. Otherwise, it 
        # finds that many gait cycles, working backwards from end of trial.
               
        # Helper functions
        def detect_gait_peaks(r_calc_rel_x,
                              l_calc_rel_x,
                              r_toe_rel_x,
                              l_toe_rel_x,
                              prominence = 0.3):
            # Find HS.
            rHS, _ = find_peaks(r_calc_rel_x, prominence=prominence)
            lHS, _ = find_peaks(l_calc_rel_x, prominence=prominence)
            
            # Find TO.
            rTO, _ = find_peaks(-r_toe_rel_x, prominence=prominence)
            lTO, _ = find_peaks(-l_toe_rel_x, prominence=prominence)
            
            return rHS,lHS,rTO,lTO
        
        def detect_correct_order(rHS, rTO, lHS, lTO):
            # checks if the peaks are in the right order
                    
            expectedOrder = {'rHS': 'lTO',
                             'lTO': 'lHS',
                             'lHS': 'rTO',
                             'rTO': 'rHS'}
                    
            # Identify vector that has the smallest value in it. Put this vector name
            # in vName1
            vectors = {'rHS': rHS, 'rTO': rTO, 'lHS': lHS, 'lTO': lTO}
            non_empty_vectors = {k: v for k, v in vectors.items() if len(v) > 0}
        
            # Check if there are any non-empty vectors
            if not non_empty_vectors:
                return True  # All vectors are empty, consider it correct order
        
            vName1 = min(non_empty_vectors, key=lambda k: non_empty_vectors[k][0])
        
            # While there are any values in any of the vectors (rHS, rTO, lHS, or lTO)
            while any([len(vName) > 0 for vName in vectors.values()]):
                # Delete the smallest value from the vName1
                vectors[vName1] = np.delete(vectors[vName1], 0)
        
                # Then find the vector with the next smallest value. Define vName2 as the
                # name of this vector
                non_empty_vectors = {k: v for k, v in vectors.items() if len(v) > 0}
                
                # Check if there are any non-empty vectors
                if not non_empty_vectors:
                    break  # All vectors are empty, consider it correct order
        
                vName2 = min(non_empty_vectors, key=lambda k: non_empty_vectors[k][0])
        
                # If vName2 != expectedOrder[vName1], return False
                if vName2 != expectedOrder[vName1]:
                    return False
        
                # Set vName1 equal to vName2 and clear vName2
                vName1, vName2 = vName2, ''
        
            return True
        
        # Subtract sacrum from foot.
        # It looks like the position-based approach will be more robust.        
        r_calc_rel = (
            self.markerDict['markers']['r_calc_study'] - 
            self.markerDict['markers']['r.PSIS_study'])
        
        r_toe_rel = (
            self.markerDict['markers']['r_toe_study'] - 
            self.markerDict['markers']['r.PSIS_study'])
        r_toe_rel_x = r_toe_rel[:,0]
        # Repeat for left.
        l_calc_rel = (
            self.markerDict['markers']['L_calc_study'] - 
            self.markerDict['markers']['L.PSIS_study'])
        l_toe_rel = (
            self.markerDict['markers']['L_toe_study'] - 
            self.markerDict['markers']['L.PSIS_study'])
        
        # Identify which direction the subject is walking.
        mid_psis = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study'])/2
        mid_asis = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study'])/2
        mid_dir = mid_asis - mid_psis
        mid_dir_floor = np.copy(mid_dir)
        mid_dir_floor[:,1] = 0
        mid_dir_floor = mid_dir_floor / np.linalg.norm(mid_dir_floor,axis=1,keepdims=True)
        
        # Dot product projections   
        r_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor,r_calc_rel)
        l_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor,l_calc_rel)
        r_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor,r_toe_rel)
        l_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor,l_toe_rel)
        
        # Old Approach that does not take the heading direction into account.
        # r_psis_x = self.markerDict['markers']['r.PSIS_study'][:,0]
        # r_asis_x = self.markerDict['markers']['r.ASIS_study'][:,0]
        # r_dir_x = r_asis_x-r_psis_x
        # position_approach_scaling = np.where(r_dir_x > 0, 1, -1)        
        # r_calc_rel_x = r_calc_rel[:,0] * position_approach_scaling
        # r_toe_rel_x = r_toe_rel[:,0] * position_approach_scaling
        # l_calc_rel_x = l_calc_rel[:,0] * position_approach_scaling
        # l_toe_rel_x = l_toe_rel[:,0] * position_approach_scaling
                       
        # Detect peaks, check if they're in the right order, if not reduce prominence.
        # the peaks can be less prominent with pathological or slower gait patterns
        prominences = [0.3, 0.25, 0.2]
        
        for i,prom in enumerate(prominences):
            rHS,lHS,rTO,lTO = detect_gait_peaks(r_calc_rel_x=r_calc_rel_x,
                                  l_calc_rel_x=l_calc_rel_x,
                                  r_toe_rel_x=r_toe_rel_x,
                                  l_toe_rel_x=l_toe_rel_x,
                                  prominence=prom)
            if not detect_correct_order(rHS=rHS, rTO=rTO, lHS=lHS, lTO=lTO):
                if prom == prominences[-1]:
                    raise ValueError('The ordering of gait events is not correct. Consider trimming your trial using the trimming_start and trimming_end options.')
                else:
                    print('The gait events were not in the correct order. Trying peak detection again ' +
                      'with prominence = ' + str(prominences[i+1]) + '.')
            else:
                # everything was in the correct order. continue.
                break
        
        if visualize:
            import matplotlib.pyplot as plt
            plt.close('all')
            plt.figure(1)
            plt.plot(self.markerDict['time'],r_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],r_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][rHS], r_calc_rel_x[rHS], color='red', label='rHS')
            plt.scatter(self.markerDict['time'][rTO], r_toe_rel_x[rTO], color='blue', label='rTO')
            plt.legend()

            plt.figure(2)
            plt.plot(self.markerDict['time'],l_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],l_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][lHS], l_calc_rel_x[lHS], color='red', label='lHS')
            plt.scatter(self.markerDict['time'][lTO], l_toe_rel_x[lTO], color='blue', label='lTO')
            plt.legend()

        # Find the number of gait cycles for the foot of interest.
        if leg=='auto':
            # Find the last HS of either foot.
            if rHS[-1] > lHS[-1]:
                leg = 'r'
            else:
                leg = 'l'
        
        # Find the number of gait cycles for the foot of interest.
        if leg == 'r':
            hsIps = rHS
            toIps = rTO
            hsCont = lHS
            toCont = lTO
        elif leg == 'l':
            hsIps = lHS
            toIps = lTO
            hsCont = rHS
            toCont = rTO

        if len(hsIps)-1 < n_gait_cycles:
            print('You requested {} gait cycles, but only {} were found. '
                  'Proceeding with this number.'.format(n_gait_cycles,len(hsIps)-1))
            n_gait_cycles = len(hsIps)-1
        if n_gait_cycles == -1:
            n_gait_cycles = len(hsIps)-1
            print('Processing {} gait cycles, leg: '.format(n_gait_cycles) + leg + '.')
            
        # Ipsilateral gait events: heel strike, toe-off, heel strike.
        gaitEvents_ips = np.zeros((n_gait_cycles, 3),dtype=int)
        # Contralateral gait events: toe-off, heel strike.
        gaitEvents_cont = np.zeros((n_gait_cycles, 2),dtype=int)
        if n_gait_cycles <1:
            raise Exception('Not enough gait cycles found.')

        for i in range(n_gait_cycles):
            # Ipsilateral HS, TO, HS.
            gaitEvents_ips[i,0] = hsIps[-i-2]
            gaitEvents_ips[i,2] = hsIps[-i-1]
            
            # Iterate in reverse through ipsilateral TO, finding the one that
            # is within the range of gaitEvents_ips.
            toIpsFound = False
            for j in range(len(toIps)):
                if toIps[-j-1] > gaitEvents_ips[i,0] and toIps[-j-1] < gaitEvents_ips[i,2] and not toIpsFound:
                    gaitEvents_ips[i,1] = toIps[-j-1]
                    toIpsFound = True

            # Contralateral TO, HS.
            # Iterate in reverse through contralateral HS and TO, finding the
            # one that is within the range of gaitEvents_ips
            hsContFound = False
            toContFound = False
            for j in range(len(toCont)):
                if toCont[-j-1] > gaitEvents_ips[i,0] and toCont[-j-1] < gaitEvents_ips[i,2] and not toContFound:
                    gaitEvents_cont[i,0] = toCont[-j-1]
                    toContFound = True
                    
            for j in range(len(hsCont)):
                if hsCont[-j-1] > gaitEvents_ips[i,0] and hsCont[-j-1] < gaitEvents_ips[i,2] and not hsContFound:
                    gaitEvents_cont[i,1] = hsCont[-j-1]
                    hsContFound = True
            
            # Skip this step if no contralateral peaks fell within ipsilateral events
            # This can happen with noisy data with subject far from camera. 
            if not toContFound or not hsContFound:                   
                print('Could not find contralateral gait event within ' + 
                               'ipsilateral gait event range ' + str(i+1) + 
                               ' steps until the end. Skipping this step.')
                gaitEvents_cont[i,:] = -1
                gaitEvents_ips[i,:] = -1
        
        # Remove any nan rows
        mask_ips = (gaitEvents_ips == -1).any(axis=1)
        if all(mask_ips):
            raise Exception('No good steps for ' + leg + ' leg.')
        gaitEvents_ips = gaitEvents_ips[~mask_ips]
        gaitEvents_cont = gaitEvents_cont[~mask_ips]
            
        # Convert gaitEvents to times using self.markerDict['time'].
        gaitEventTimes_ips = self.markerDict['time'][gaitEvents_ips]
        gaitEventTimes_cont = self.markerDict['time'][gaitEvents_cont]
                            
        gaitEvents = {'ipsilateralIdx':gaitEvents_ips,
                      'contralateralIdx':gaitEvents_cont,
                      'ipsilateralTime':gaitEventTimes_ips,
                      'contralateralTime':gaitEventTimes_cont,
                      'eventNamesIpsilateral':['HS','TO','HS'],
                      'eventNamesContralateral':['TO','HS'],
                      'ipsilateralLeg':leg}
        
        return gaitEvents