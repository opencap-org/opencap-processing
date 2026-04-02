"""
    ---------------------------------------------------------------------------
    OpenCap processing: stsAnalysis.py
    ---------------------------------------------------------------------------

    Copyright TBD

    Author(s): Antoine Falisse, Scott Uhlrich, RD Magruder

    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

from utilsKinematics import kinematics


class sts_analysis(kinematics):
    def __init__(self, session_dir, trial_name, leg='auto',
                    lowpass_cutoff_frequency_for_coordinate_values=6,
                 n_sts_cycles=-1, trimming_start = 0, trimming_end = 0):
        super().__init__(session_dir,
                         trial_name,
                         lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)

        self.trimming_start = trimming_start
        self.trimming_end = trimming_end

        self.markerDict = self.get_marker_dict(session_dir, trial_name,
                                               lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values)

        self.coordinateValues = self.get_coordinate_values()

        if self.trimming_start > 0:
            self.idx_trim_start = np.where(np.round(self.markerDict['time'] - self.trimming_start, 6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:, ]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:, :]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]

        if self.trimming_end > 0:
            self.idx_trim_end = np.where(
                np.round(self.markerDict['time'], 6) <= np.round(self.markerDict['time'][-1] - self.trimming_end, 6))[
                                    0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end, ]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end, :]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]

        # Currently not being used
        self.rotation_about_y, self.markerDictRotated = self.rotate_x_forward()

        # Get body angular position and velocity in ground frame
        self.body_angles = self.get_body_orientation(expressed_in='ground',
                                                          lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values)
        self.body_angular_velocity = self.get_body_angular_velocity(expressed_in='ground',
                                                                    lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values)

        self.stsEvents = self.segment_sts(n_sts_cycles=n_sts_cycles)
        self.n_sts_cycles = np.shape(self.stsEvents['startRisingIdx'])[0]

        self.sts_kinetics = None # Placeholder for kinetics data

    def rotate_x_forward(self):
        # Find the midpoint of the PSIS markers
        psis_midpoint = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study']) / 2

        # Find the midpoint of the ASIS markers
        asis_midpoint = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study']) / 2

        # Compute the vector pointing from the PSIS midpoint to the ASIS midpoint
        heading_vector = asis_midpoint - psis_midpoint

        # Compute the angle between the heading vector projected onto x-z plane and x-axis
        angle = np.unwrap(np.arctan2(heading_vector[:, 2], heading_vector[:, 0]))

        # compute average angle during middle 50% of the trial
        n_frames = len(self.markerDict['time'])
        start_index = int(n_frames * 0.25)
        end_index = int(n_frames * 0.75)
        angle = np.degrees(np.mean(angle[start_index:end_index], axis=0))

        # Apply the rotation to the marker data
        marker_dict_rotated = self.rotate_marker_dict(self.markerDict, {'y': angle})

        return angle, marker_dict_rotated


    def segment_sts(self, n_sts_cycles=-1, velSeated=0.1, velStanding=0.1,
                    visualize=False, delay=0.1, lean_threshold=-0.05):
        # Extract pelvis height and time
        pelvis_ty = self.coordinateValues['pelvis_ty']
        timeVec = self.markerDict['time']
        dt = timeVec[1] - timeVec[0]

        # Extract torso lean angle and angular velocity
        torso_z = self.body_angles['torso_z']
        torso_z_vel = self.body_angular_velocity['torso_z']

        # Normalize pelvis height signal
        pelvSignal = np.array(pelvis_ty - np.min(pelvis_ty))
        pelvVel = np.diff(pelvSignal, append=0) / dt

        # Find peaks in pelvis vertical position (STS max points)
        idxMaxPelvTy, _ = signal.find_peaks(pelvSignal - np.min(pelvSignal), distance=.9 / dt, height=.12, prominence=.12)

        # Initialize storage
        maxIdxOld = 0
        startFinishInds = []
        forwardLeanInds = []

        for i, maxIdx in enumerate(idxMaxPelvTy):
            # Identify velocity peak before pelvis peak
            vels = pelvVel[maxIdxOld:maxIdx]
            velPeak, peakVals = signal.find_peaks(vels, distance=.9 / dt, height=.2)
            velPeak = velPeak[np.argmax(peakVals['peak_heights'])] + maxIdxOld

            # Find sitting and standing transitions
            velsLeftOfPeak = np.flip(pelvVel[maxIdxOld:velPeak])
            velsRightOfPeak = pelvVel[velPeak:]

            slowingIndLeft = np.argwhere(velsLeftOfPeak < velSeated)[0]
            startIdx = velPeak - slowingIndLeft
            slowingIndRight = np.argwhere(velsRightOfPeak < velStanding)[0]
            endIdx = velPeak + slowingIndRight

            startFinishInds.append([startIdx[0], endIdx[0]])

            # Define a temporary seating index for better forward lean detection
            if i == 0:
                temp_sit_ind = maxIdxOld
            else:
                temp_sit_ind = maxIdxOld + np.argmin(pelvVel[maxIdxOld:startIdx[0]])

            # Detect forward lean onset
            torso_vel_segment = torso_z_vel[int(temp_sit_ind):int(startIdx[0])]  # Before lift-off
            torso_vel_segment = np.flip(torso_vel_segment)  # Reverse for backward search

            # Find the first time angular velocity is below the threshold (leaning forward)
            lean_indices = np.argwhere(torso_vel_segment > lean_threshold)
            if lean_indices.size > 0:
                forwardLeanIdx = startIdx[0] - lean_indices[0][0]  # Convert back to original indexing
            else:
                forwardLeanIdx = startIdx[0]  # Default to lift-off if no lean detected

            forwardLeanInds.append(forwardLeanIdx)
            maxIdxOld = np.copy(maxIdx)

        # Convert times
        risingTimes = [timeVec[i].tolist() for i in startFinishInds]
        forwardLeanTimes = [timeVec[i].tolist() for i in forwardLeanInds]

        # Apply delay correction
        sf = 1 / np.round(np.mean(np.round(timeVec[1:] - timeVec[:-1], 2)), 16)
        startFinishIndsDelay = []
        for i in startFinishInds:
            c_i = []
            for c_j, j in enumerate(i):
                if c_j == 0:
                    c_i.append(j + int(delay * sf))
                else:
                    c_i.append(j)
            startFinishIndsDelay.append(c_i)
        risingTimesDelayedStart = [
            timeVec[i].tolist() for i in startFinishIndsDelay]

        # Adjust for periodicity
        startFinishIndsDelayPeriodic = []
        for i, val in enumerate(startFinishIndsDelay):
            pelvVal_up = pelvSignal[val[0]]
            val_down = np.argwhere(pelvSignal[val[1] + 1:] < (pelvVal_up+0.05))[0][0] + val[1] + 1 # 5cm threshold above where the pelvis started rising

            original_val_down = val_down
            last_index = startFinishIndsDelay[i+1][1] if i < len(startFinishIndsDelay) - 1 else len(pelvSignal) - 1
            while val_down < last_index:   # Don't go past end of trial
                if pelvSignal[val_down] <= pelvVal_up or pelvVel[val_down] >= 0:
                    break
                val_down += 1


            if val_down >= last_index:
                val_down = original_val_down
            else:
                val_down -=1
            startFinishIndsDelayPeriodic.append([val[0], val_down])

        risingSittingTimesDelayedStartPeriodicEnd = [timeVec[i].tolist() for i in startFinishIndsDelayPeriodic]

        if visualize:
            plt.figure()
            plt.plot(pelvSignal)
            for c_v, val in enumerate(startFinishInds):
                plt.plot(val, pelvSignal[val], marker='o', markerfacecolor='k',
                         markeredgecolor='none', linestyle='none',
                         label='Rising phase')
                val2 = startFinishIndsDelay[c_v][0]
                plt.plot(val2, pelvSignal[val2], marker='o',
                         markerfacecolor='r', markeredgecolor='none',
                         linestyle='none', label='Delayed start')
                val3 = startFinishIndsDelayPeriodic[c_v][1]
                plt.plot(val3, pelvSignal[val3], marker='o',
                         markerfacecolor='g', markeredgecolor='none',
                         linestyle='none',
                         label='Periodic end corresponding to delayed start')
                val4 = forwardLeanInds[c_v]
                plt.plot(val4, pelvSignal[val4], marker='o',
                         markerfacecolor='b', markeredgecolor='none',
                         linestyle='none', label='Forward Lean Start')
                if c_v == 0:
                    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.40), ncol=2)
            plt.xlabel('Frames')
            plt.ylabel('Position [m]')
            plt.title('Vertical pelvis position')
            plt.tight_layout()
            plt.show()

        # Ensure correct STS cycle count
        actual_cycles = len(risingSittingTimesDelayedStartPeriodicEnd)
        if actual_cycles < n_sts_cycles or n_sts_cycles == -1:
            n_sts_cycles = actual_cycles

        if n_sts_cycles < 1:
            raise Exception('No STS cycles found.')
        else:
            print('Found and Processing', n_sts_cycles, 'STS cycles.')

        # Build output dictionary
        stsEvents = {
            'startRisingIdx': [startFinishIndsDelay[i][0] for i in range(n_sts_cycles)],
            'startRisingTime': [risingTimesDelayedStart[i][0] for i in range(n_sts_cycles)],
            'endRisingIdx': [startFinishInds[i][1] for i in range(n_sts_cycles)],
            'endRisingTime': [risingTimes[i][1] for i in range(n_sts_cycles)],
            'sittingIdx': [startFinishIndsDelayPeriodic[i][1] for i in range(n_sts_cycles)],
            'sittingTime': [risingSittingTimesDelayedStartPeriodicEnd[i][1] for i in range(n_sts_cycles)],
            'forwardLeanIdx': [forwardLeanInds[i] for i in range(n_sts_cycles)],
            'forwardLeanTime': [forwardLeanTimes[i] for i in range(n_sts_cycles)]
        }

        return stsEvents


    def compute_scalars(self, scalarNames, return_all=False):

        # Verify that scalarNames are methods in gait_analysis.
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if 'compute_' in entry]

        if scalarNames is None:
            print('No scalars defined, these methods are available:')
            print(*possibleMethods)
            return

        nonexistant_methods = [entry for entry in scalarNames if 'compute_' + entry not in method_names]

        if len(nonexistant_methods) > 0:
            raise Exception(
                str(['compute_' + a for a in nonexistant_methods]) + ' does not exist in sts_analysis class.')

        scalarDict = {}
        for scalarName in scalarNames:
            thisFunction = getattr(self, 'compute_' + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]['value'],
             scalarDict[scalarName]['units']) = thisFunction(return_all=return_all)

        return scalarDict

    # Find the maximum torso angle after lean onset, prior to lift-off
    def compute_torso_orientation(self, return_all=False):
        # Extract torso lean angle
        torso_z = self.body_angles['torso_z'] * -1 # Invert to match the definition of forward lean

        max_torso_leans = []
        for i, val in enumerate(self.stsEvents['forwardLeanIdx']):
            max_torso_leans.append(np.max(torso_z[val:self.stsEvents['startRisingIdx'][i]]))

        # Convert to degrees
        max_torso_leans = np.degrees(max_torso_leans)
        units = 'degrees'

        max_torso_lean = np.mean(max_torso_leans)

        if return_all:
            return max_torso_leans, units
        else:
            return max_torso_lean, units

    # Find the torso angle at lift-off
    def compute_torso_orientation_liftoff(self, return_all=False):
        # Extract torso lean angle
        torso_z = self.body_angles['torso_z'] * -1 # Invert to match the definition of forward lean

        torso_lean_liftoffs = []
        for i, val in enumerate(self.stsEvents['startRisingIdx']):
            torso_lean_liftoffs.append(torso_z[val])

        # Convert to degrees
        torso_lean_liftoffs = np.degrees(torso_lean_liftoffs)
        units = 'degrees'

        torso_lean_liftoff = np.mean(torso_lean_liftoffs)

        if return_all:
            return torso_lean_liftoffs, units
        else:
            return torso_lean_liftoff, units

    # Find the maximum torso angular velocity after lean onset, prior to lift-off
    def compute_torso_angular_velocity(self, return_all=False):
        # Extract torso angular velocity
        torso_z_vel = self.body_angular_velocity['torso_z'] * -1 # Invert to match the definition of forward lean

        # Find the maximum torso angular velocity after lean onset, prior to lift-off
        max_torso_vels = []
        for i, val in enumerate(self.stsEvents['forwardLeanIdx']):
            max_torso_vels.append(np.max(torso_z_vel[val:self.stsEvents['startRisingIdx'][i]]))

        # Convert to degrees
        max_torso_vels = np.degrees(max_torso_vels)
        units = 'degrees/s'

        max_torso_vel = np.mean(max_torso_vels)

        if return_all:
            return max_torso_vels, units
        else:
            return max_torso_vel, units

    # Find the torso angular velocity at lift-off
    def compute_torso_angular_velocity_liftoff(self, return_all=False):
        # Extract torso angular velocity
        torso_z_vel = self.body_angular_velocity['torso_z'] * -1 # Invert to match the definition of forward lean

        # Find the torso angular velocity at lift-off
        torso_vel_liftoffs = []
        for i, val in enumerate(self.stsEvents['startRisingIdx']):
            torso_vel_liftoffs.append(torso_z_vel[val])

        # Convert to degrees
        torso_vel_liftoffs = np.degrees(torso_vel_liftoffs)
        units = 'degrees/s'

        torso_vel_liftoff = np.mean(torso_vel_liftoffs)

        if return_all:
            return torso_vel_liftoffs, units
        else:
            return torso_vel_liftoff, units

    # Find how long it takes to stand up
    def compute_rise_time(self, return_all=False):
        # Extract time
        timeVec = self.markerDict['time']

        # Compute rise time
        rise_times = []
        for i, val in enumerate(self.stsEvents['startRisingIdx']):
            rise_times.append(timeVec[self.stsEvents['endRisingIdx'][i]] - timeVec[val])

        units = 's'

        rise_time = np.mean(rise_times)

        if return_all:
            return rise_times, units
        else:
            return rise_time, units

    # Find how long it takes to stand up and sit down
    # NOTE: Sitting time is delayed rising time to sitting, where sitting is 10cm prior to touching the chair
    def compute_sts_time(self, return_all=False):
        # Extract time
        timeVec = self.markerDict['time']

        # Compute sit-to-stand time
        sts_times = []
        for i, val in enumerate(self.stsEvents['startRisingIdx']):
            sts_times.append(timeVec[self.stsEvents['sittingIdx'][i]] - timeVec[val])

        units = 's'

        sts_time = np.mean(sts_times)

        if return_all:
            return sts_times, units
        else:
            return sts_time, units

    # Find the maximum knee extention moment during sit-to-stand
    def compute_max_knee_extension_moment(self, return_all=False):
        if self.sts_kinetics is None:
            raise Exception('No kinetics data available. Run run_dynamic_simulation() first.')

        # Find the maximum knee extension moment during sit-to-stand
        max_knee_l_moments = []
        max_knee_r_moments = []
        for i, sts_kinetics in enumerate(self.sts_kinetics):
            knee_l_idx = sts_kinetics.optimal_result["torque_labels"].index("knee_angle_l_moment")
            knee_r_idx = sts_kinetics.optimal_result["torque_labels"].index("knee_angle_r_moment")

            endTime = np.searchsorted(sts_kinetics.time,self.stsEvents['endRisingTime'][4])
            knee_l_moments = sts_kinetics.optimal_result["torques"][knee_l_idx, :endTime] * -1 # Invert to match the definition of knee extension
            knee_r_moments = sts_kinetics.optimal_result["torques"][knee_r_idx, :endTime] * -1 # Invert to match the definition of knee extension

            if knee_r_moments.size == 0 or knee_l_moments.size == 0:
                continue

            max_knee_l_moments.append(np.max(knee_l_moments))
            max_knee_r_moments.append(np.max(knee_r_moments))

        units = 'N*m'

        max_knee_l_moment = np.nanmean(max_knee_l_moments) # Excludes trials when no moment is found
        max_knee_r_moment = np.nanmean(max_knee_r_moments) # Excludes trials when no moment is found

        if return_all:
            return (max_knee_l_moments, max_knee_r_moments), units
        else:
            return (max_knee_l_moment, max_knee_r_moment), units

    # Run an inverse dynamics simulation for sit-to-stand
    def run_dynamic_simulation(self, baseDir, dataFolder, session_id, trial_name, case='0', repetition=-1, verbose=False):
        # Import Dynamics as necessary
        import sys
        sys.path.append('../UtilsDynamicSimulations/OpenSimAD')

        from UtilsDynamicSimulations.OpenSimAD.utilsOpenSimAD import processInputsOpenSimAD
        from UtilsDynamicSimulations.OpenSimAD.mainOpenSimAD import run_tracking
        from UtilsDynamicSimulations.OpenSimAD.utilsKineticsOpenSimAD import kineticsOpenSimAD

        def plot_kinetics(sts, sts_kinetics_all_cycles, torque_label_l_r, title):
            # Create subplots for pelvis_ty and the two knee moments
            fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

            # Plot Pelvis Ty (first subplot)
            time_vec = sts.coordinateValues["time"]
            pelvis_ty = sts.coordinateValues["pelvis_ty"]
            sns.lineplot(x=time_vec, y=pelvis_ty, ax=axs[0], label="Pelvis Ty", color='black')
            axs[0].set_ylabel('Pelvis Ty')
            axs[0].legend()

            # Plot knee moments for each cycle in the second subplot (left knee moment)
            for i, sts_kinetics in enumerate(sts_kinetics_all_cycles):
                # Extract time and torque data
                time_vec_cycle = sts_kinetics.time
                knee_l_idx = sts_kinetics.optimal_result["torque_labels"].index(torque_label_l_r[0])
                knee_r_idx = sts_kinetics.optimal_result["torque_labels"].index(torque_label_l_r[1])

                # Extract torques
                knee_l_torque = sts_kinetics.optimal_result["torques"][knee_l_idx, :]
                knee_r_torque = sts_kinetics.optimal_result["torques"][knee_r_idx, :]

                # Plot the left knee torque in the second subplot
                sns.lineplot(x=time_vec_cycle, y=knee_l_torque, ax=axs[1], label=f"Cycle {i + 1}")
                # Plot the right knee torque in the third subplot
                sns.lineplot(x=time_vec_cycle, y=knee_r_torque, ax=axs[2], label=f"Cycle {i + 1}")

            # Set labels and title for each plot
            axs[1].set_ylabel('Knee L Moment')
            axs[2].set_ylabel('Knee R Moment')
            axs[2].set_xlabel("Time (s)")
            axs[0].set_title(title)

            # Add legends to each plot
            axs[1].legend()
            axs[2].legend()

            plt.tight_layout()
            plt.show()

        if repetition == -1:
            sts_kinetics_all_cycles = []  # List to store kinetics for all cycles
            for i in range(self.n_sts_cycles):
                try:
                    sts_kinetics = kineticsOpenSimAD(dataFolder, session_id, trial_name, case=case, repetition=i)
                except:
                    # Run tracking if the result doesn't exist
                    settings = processInputsOpenSimAD(baseDir, dataFolder, session_id, trial_name,
                                                      'sit_to_stand', repetition=i)
                    run_tracking(baseDir, dataFolder, session_id, settings, case=case)
                    sts_kinetics = kineticsOpenSimAD(dataFolder, session_id, trial_name, case=case, repetition=i)

                # Append the current cycle's kinetics to the list
                sts_kinetics_all_cycles.append(sts_kinetics)
            self.sts_kinetics = sts_kinetics_all_cycles
        else:
            try:  # Try to load results for the specific repetition.
                sts_kinetics = kineticsOpenSimAD(dataFolder, session_id, trial_name, case=case, repetition=repetition)
            except:  # If results do not exist, run the simulation.
                settings = processInputsOpenSimAD(baseDir, dataFolder, session_id, trial_name,
                                                  'sit_to_stand', repetition=repetition)
                run_tracking(baseDir, dataFolder, session_id, settings, case=case)
                sts_kinetics = kineticsOpenSimAD(dataFolder, session_id, trial_name, case=case, repetition=repetition)

            self.sts_kinetics = [sts_kinetics]
        # Plot all cycles
        if verbose:
            plot_kinetics(self, self.sts_kinetics, ["knee_angle_l_moment", "knee_angle_r_moment"],
                          "STS Knee Extension Moments")

        return self.sts_kinetics