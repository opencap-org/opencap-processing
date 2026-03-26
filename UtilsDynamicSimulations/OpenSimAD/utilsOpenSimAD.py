'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsOpenSimAD.py
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
    
    This script contains helper utilities.
'''

import os
import sys
import opensim
import numpy as np
import pandas as pd
import casadi as ca
import shutil
import importlib
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import platform
import urllib.request
import requests
import zipfile
import seaborn as sns
import subprocess
import re

from utils import (storage_to_numpy, storage_to_dataframe, 
                   download_kinematics, import_metadata, numpy_to_storage)
from utilsProcessing import (segment_squats, segment_STS, adjust_muscle_wrapping,
                             generate_model_with_contacts)
from settingsOpenSimAD import get_setup

from squat_analysis import squat_analysis

# %% Filter numpy array.
def filterNumpyArray(array, time, cutoff_frequency=6, order=4):
    
    fs = np.round(1 / np.mean(np.diff(time)), 6)
    fc = cutoff_frequency
    w = fc / (fs / 2)
    b, a = signal.butter(order/2, w, 'low')  
    arrayFilt = signal.filtfilt(
        b, a, array, axis=0, 
        padtype='odd', padlen=3*(max(len(b),len(a))-1))    
    # print('numpy array filtered at {}Hz.'.format(cutoff_frequency)) 
    
    return arrayFilt

# %% Interpolate numpy array.
def interpolateNumpyArray_time(data, time, tIn, tEnd, N): 

    tOut = np.linspace(tIn, tEnd, N)
    if data.ndim == 1:
        set_interp = interp1d(time, data)
        dataInterp = set_interp(tOut)
    else:
        dataInterp = np.zeros((N, data.shape[1]))
        for i in range(data.shape[1]):
            set_interp = interp1d(time, data[:, i])
            dataInterp[:, i] = set_interp(tOut)
            
    return dataInterp 

# %% Solve problem with bounds instead of constraints.
def solve_with_bounds(opti, tolerance, useExpressionGraphFunction):
    
    # Get guess.
    guess = opti.debug.value(opti.x, opti.initial())
    # Sparsity pattern of the constraint Jacobian.
    jac = ca.jacobian(opti.g, opti.x)
    sp = (ca.DM(jac.sparsity(), 1)).sparse()
    # Find constraints dependent on one variable.
    is_single = np.sum(sp, axis=1)
    is_single_num = np.zeros(is_single.shape[0])
    for i in range(is_single.shape[0]):
        is_single_num[i] = np.equal(is_single[i, 0], 1)
    # Find constraints with linear dependencies or no dependencies.
    is_nonlinear = ca.which_depends(opti.g, opti.x, 2, True)
    is_linear = [not i for i in is_nonlinear]
    is_linear_np = np.array(is_linear)
    is_linear_np_num = is_linear_np*1
    # Constraints dependent linearly on one variable should become bounds.
    is_simple = is_single_num.astype(int) & is_linear_np_num
    idx_is_simple = [i for i, x in enumerate(is_simple) if x]
    # Find corresponding variables.
    col = np.nonzero(sp[idx_is_simple, :].T)[0]
    # Contraint values.
    lbg = opti.lbg
    lbg = opti.value(lbg)
    ubg = opti.ubg
    ubg = opti.value(ubg)
    # Detect  f2(p)x+f1(p)==0
    # This is important if you have scaled variables: x = 10*opti.variable()
    # with a constraint -10 < x < 10. Because in the reformulation we read out 
    # the original variable and thus we need to scale the bounds appropriately.
    g = opti.g
    gf = ca.Function('gf', [opti.x, opti.p], [g[idx_is_simple, 0], 
                            ca.jtimes(g[idx_is_simple, 0], opti.x, 
                                      np.ones((opti.nx, 1)))])
    [f1, f2] = gf(0, opti.p)
    f1 = (ca.evalf(f1)).full()
    f2 = (ca.evalf(f2)).full()
    lb = (lbg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    ub = (ubg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    # Initialize bound vector.
    lbx = -np.inf * np.ones((opti.nx))
    ubx = np.inf * np.ones((opti.nx))
    # Fill bound vector. For unbounded variables, we keep +/- inf.
    for i in range(col.shape[0]):
        lbx[col[i]] = np.maximum(lbx[col[i]], lb[i])
        ubx[col[i]] = np.minimum(ubx[col[i]], ub[i])      
    lbx[col] = (lbg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    ubx[col] = (ubg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    # Updated constraint value vector.
    not_idx_is_simple = np.delete(range(0, is_simple.shape[0]), idx_is_simple)
    new_g = g[not_idx_is_simple, 0]
    # Updated bounds.
    llb = lbg[not_idx_is_simple]
    uub = ubg[not_idx_is_simple]
    
    prob = {'x': opti.x, 'f': opti.f, 'g': new_g}
    s_opts = {}
    if useExpressionGraphFunction:
        s_opts["expand"] = True
    else:
        s_opts["expand"] = False
    s_opts["ipopt.hessian_approximation"] = "limited-memory"
    s_opts["ipopt.mu_strategy"] = "adaptive"
    s_opts["ipopt.max_iter"] = 5000
    s_opts["ipopt.tol"] = 10**(-tolerance)
    solver = ca.nlpsol("solver", "ipopt", prob, s_opts)
    # Solve.
    arg = {}
    arg["x0"] = guess
    # Bounds on x.
    arg["lbx"] = lbx
    arg["ubx"] = ubx
    # Bounds on g.
    arg["lbg"] = llb
    arg["ubg"] = uub    
    sol = solver(**arg) 
    # Extract and save results.
    w_opt = sol['x'].full()
    stats = solver.stats()
    
    return w_opt, stats

# %% Solver problem with constraints and not bounds.
def solve_with_constraints(opti, tolerance):
    
    s_opts = {"hessian_approximation": "limited-memory",
              "mu_strategy": "adaptive",
              "max_iter": 5000,
              "tol": 10**(-tolerance)}
    p_opts = {"expand":False}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()  
    
    return sol

# %% Helper plotting tools.
def plotVSBounds(y,lb,ub,title=''):
    
    ny = np.ceil(np.sqrt(y.shape[0]))   
    fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
    fig.suptitle(title)
    x = np.linspace(1,y.shape[1],y.shape[1])
    for i, ax in enumerate(axs.flat):
        if i < y.shape[0]:
            ax.plot(x,y[i,:],'k')
            ax.hlines(lb[i,0],x[0],x[-1],'r')
            ax.hlines(ub[i,0],x[0],x[-1],'b')
    plt.show()
            
def plotVSvaryingBounds(y,lb,ub,title=''):
    
    ny = np.ceil(np.sqrt(y.shape[0]))   
    fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
    fig.suptitle(title)
    x = np.linspace(1,y.shape[1],y.shape[1])
    for i, ax in enumerate(axs.flat):
        if i < y.shape[0]:
            ax.plot(x,y[i,:],'k')
            ax.plot(x,lb[i,:],'r')
            ax.plot(x,ub[i,:],'b')
    plt.show()

# %% Helper function.
def getColfromk(xk, d, N):
    
    xj = np.ones((1, d*N))
    count = 0
    for k in range(N):
        for c in range(d):
            xj[0,count] = xk[0,k]
            count += 1
            
    return xj

# %% Verify if within range used for fitting polynomials.
def checkQsWithinPolynomialBounds(data, bounds, model_bounds, coordinates):
    
    updated_bounds = {}    
    for coord in coordinates:
        if coord in bounds:
            c_idc = coordinates.index(coord)
            c_data = data[c_idc, :]
            # Small margin to account for filtering.                
            if not np.all(c_data * 180 / np.pi <= bounds[coord]['max']):
                print('WARNING: the {} coordinate values to track have values above the default upper bound ROM for polynomial fitting: {}deg >= {}deg'.format(coord, np.round(np.max(c_data) * 180 / np.pi, 2), np.round(bounds[coord]['max'], 2)))
                new_bound = np.ceil(np.max(c_data) * 180 / np.pi)
                if new_bound > model_bounds[coord]['max']:
                    print('The maximal value is above the model upper bound, this might happen when filtering data. The model upper bound will be used instead.')
                    new_bound = model_bounds[coord]['max']
                print('Upper bound set to: {}deg'.format(new_bound))
                updated_bounds[coord] = {'max': new_bound}
            if not np.all(c_data * 180 / np.pi >= bounds[coord]['min']):
                print('WARNING: the {} coordinate values to track have values below default lower bound ROM for polynomial fitting: {}deg <= {}deg'.format(coord, np.round(np.min(c_data) * 180 / np.pi, 2), np.round(bounds[coord]['min'], 2)))
                new_bound = np.floor(np.min(c_data) * 180 / np.pi)
                if new_bound < model_bounds[coord]['min']:
                    print('The minimal value is below the model lower bound, this might happen when filtering data. The model lower bound will be used instead.')
                    new_bound = model_bounds[coord]['min']                
                if coord in updated_bounds and 'max' in updated_bounds[coord]:
                    updated_bounds[coord]['min'] = new_bound
                else:
                    updated_bounds[coord] = {'min': new_bound}
    
    return updated_bounds

# %% Extract data frame from storage file.
def getFromStorage(storage_file, headers):
    
    data = storage_to_numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% Extract EMG.
def getEMG(storage_file, headers):

    data = storage_to_numpy(storage_file)
    EMGs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        c_header = header + '_activation'
        if c_header in list(data.dtype.names):
            EMGs.insert(count + 1, header, data[c_header])
        else:
            EMGs.insert(count + 1, header, np.nan)            
    
    return EMGs

# %% Extract ID.
def getID(storage_file, headers):
    
    data = storage_to_numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        if ((header == 'pelvis_tx') or (header == 'pelvis_ty') or 
            (header == 'pelvis_tz')):
            out.insert(count + 1, header, data[header + '_force'])              
        else:
            out.insert(count + 1, header, data[header + '_moment'])    
    
    return out

# %% Extract GRF.
def getGRFAll(pathGRFFile, timeInterval, N):
    
    GRF = {        
        'headers': {
            'forces': {
                'right': ['R_ground_force_vx', 'R_ground_force_vy', 
                          'R_ground_force_vz'],
                'left': ['L_ground_force_vx', 'L_ground_force_vy', 
                         'L_ground_force_vz'],
                'all': ['R_ground_force_vx', 'R_ground_force_vy', 
                        'R_ground_force_vz','L_ground_force_vx', 
                        'L_ground_force_vy', 'L_ground_force_vz']},
            'COP': {
                'right': ['R_ground_force_px', 'R_ground_force_py', 
                          'R_ground_force_pz'],
                'left': ['L_ground_force_px', 'L_ground_force_py', 
                         'L_ground_force_pz'],
                'all': ['R_ground_force_px', 'R_ground_force_py', 
                        'R_ground_force_pz','L_ground_force_px', 
                        'L_ground_force_py', 'L_ground_force_pz']},
            'torques': {
                'right': ['R_ground_torque_x', 'R_ground_torque_y', 
                          'R_ground_torque_z'],
                'left': ['L_ground_torque_x', 'L_ground_torque_y', 
                         'L_ground_torque_z'],
                'all': ['R_ground_torque_x', 'R_ground_torque_y', 
                        'R_ground_torque_z', 'L_ground_torque_x', 
                        'L_ground_torque_y', 'L_ground_torque_z']}}}
    
    # Here we extract the GRFs and compute the GRMs wrt the ground origin.        
    GRF['df'] = {
        'forces': {
            'right':getGRF(pathGRFFile, GRF['headers']['forces']['right']),
            'left': getGRF(pathGRFFile, GRF['headers']['forces']['left'])},
        'torques_G': {
            'right': getGRM_wrt_groundOrigin(
                pathGRFFile, GRF['headers']['forces']['right'], 
                GRF['headers']['COP']['right'], 
                GRF['headers']['torques']['right']),
            'left': getGRM_wrt_groundOrigin(
                pathGRFFile, GRF['headers']['forces']['left'], 
                GRF['headers']['COP']['left'], 
                GRF['headers']['torques']['left'])}}
    GRF['df_interp'] = {
        'forces': {
            'right': interpolateDataFrame(
                GRF['df']['forces']['right'], timeInterval[0], 
                timeInterval[1], N),
            'left': interpolateDataFrame(
                GRF['df']['forces']['left'], timeInterval[0], 
                timeInterval[1], N)},
        'torques_G': {
            'right': interpolateDataFrame(
                GRF['df']['torques_G']['right'], timeInterval[0], 
                timeInterval[1], N),
            'left': interpolateDataFrame(
                GRF['df']['torques_G']['left'], timeInterval[0], 
                timeInterval[1], N)}}
    # Here we concatenate left and right, and remove the duplicated time.        
    GRF['df_interp']['forces']['all'] = pd.concat(
        [GRF['df_interp']['forces']['right'], 
          GRF['df_interp']['forces']['left']], axis=1)
    GRF['df_interp']['forces']['all'] = (
        GRF['df_interp']['forces']['all'].loc[
            :,~GRF['df_interp']['forces']['all'].columns.duplicated()])        
    GRF['df_interp']['torques_G']['all'] = pd.concat(
        [GRF['df_interp']['torques_G']['right'], 
          GRF['df_interp']['torques_G']['left']], axis=1)
    GRF['df_interp']['torques_G']['all'] = (
        GRF['df_interp']['torques_G']['all'].loc[
            :,~GRF['df_interp']['torques_G']['all'].columns.duplicated()])

    return GRF

def getGRF(storage_file, headers):

    data = storage_to_numpy(storage_file)
    GRFs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        GRFs.insert(count + 1, header, data[header])    
    
    return GRFs

# %% Extract GRF peaks.
def getGRFPeaks(GRF, timeIntervals):
    
    time = GRF['df']['forces']['right']['time']
    tin = np.argwhere(np.round(time.to_numpy(),6) == np.round(timeIntervals[0],6))[0][0]
    tout = np.argwhere(np.round(time.to_numpy(),6) == np.round(timeIntervals[1],6))[0][0]
    
    sides = ['right','left']    
    GRF_peaks = {}
    for side in sides:    
        header = GRF['headers']['forces'][side][1]
        vGRF = GRF['df']['forces'][side][header]        
        GRF_peaks[side] = np.max(vGRF[tin:tout])
        
    return GRF_peaks

# %% Compute GRM with respect to ground origin.
def getGRM_wrt_groundOrigin(storage_file, fHeaders, pHeaders, mHeaders):

    data = storage_to_numpy(storage_file)
    GRFs = pd.DataFrame()    
    for count, fheader in enumerate(fHeaders):
        GRFs.insert(count, fheader, data[fheader])  
    PoAs = pd.DataFrame()    
    for count, pheader in enumerate(pHeaders):
        PoAs.insert(count, pheader, data[pheader]) 
    GRMs = pd.DataFrame()    
    for count, mheader in enumerate(mHeaders):
        GRMs.insert(count, mheader, data[mheader])  
        
    # GRT_x = PoA_y*GRF_z - PoA_z*GRF_y
    # GRT_y = PoA_z*GRF_x - PoA_z*GRF_z + T_y
    # GRT_z = PoA_x*GRF_y - PoA_y*GRF_x
    GRM_wrt_groundOrigin = pd.DataFrame(data=data['time'], columns=['time'])    
    GRM_wrt_groundOrigin.insert(1, mHeaders[0], PoAs[pHeaders[1]] * GRFs[fHeaders[2]]  - PoAs[pHeaders[2]] * GRFs[fHeaders[1]])
    GRM_wrt_groundOrigin.insert(2, mHeaders[1], PoAs[pHeaders[2]] * GRFs[fHeaders[0]]  - PoAs[pHeaders[0]] * GRFs[fHeaders[2]] + GRMs[mHeaders[1]])
    GRM_wrt_groundOrigin.insert(3, mHeaders[2], PoAs[pHeaders[0]] * GRFs[fHeaders[1]]  - PoAs[pHeaders[1]] * GRFs[fHeaders[0]])        
    
    return GRM_wrt_groundOrigin

# %% Extract COP.
def getCOP(GRF, GRM):
    
    COP = np.zeros((3, GRF.shape[1]))
    torques = np.zeros((3, GRF.shape[1]))    
    # Only divide non-zeros
    idx_nonzeros = np.argwhere(GRF[1, :] > 0)    
    COP[0, idx_nonzeros] = GRM[2, idx_nonzeros] / GRF[1, idx_nonzeros]    
    COP[2, idx_nonzeros] = -GRM[0, idx_nonzeros] / GRF[1, idx_nonzeros]    
    torques[1, :] = GRM[1, :] - COP[2, :]*GRF[0, :] + COP[0, :]*GRF[2, :]
    
    return COP, torques

# %% Select in data frame.
def selectDataFrame(dataFrame, tIn, tEnd):

    time = dataFrame['time'].to_numpy()    
    time_start = np.argwhere(time<=tIn)[-1][0]
    time_end = np.argwhere(time>=tEnd)[0][0]
    
    return dataFrame.iloc[time_start:time_end+1]

# %% Select from data frame.
def selectFromDataFrame(dataFrame, headers):
    
    dataFrame_sel = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_sel.insert(count+1, header, dataFrame[header])
        
    return dataFrame_sel

# %% Interpolate data frame.
def interpolateDataFrame(dataFrame, tIn, tEnd, N):
    
    tOut = np.linspace(np.round(tIn,6), np.round(tEnd,6), N)    
    dataInterp = pd.DataFrame() 
    for i, col in enumerate(dataFrame.columns):
        set_interp = interp1d(np.round(dataFrame['time'],6), dataFrame[col])        
        dataInterp.insert(i, col, set_interp(tOut))
        
    return dataInterp

# %% Scale data frame.
def scaleDataFrame(dataFrame, scaling, headers):
    
    dataFrame_scaled = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_scaled.insert(count+1, header, dataFrame[header] / scaling.iloc[0][header])
        
    return dataFrame_scaled

# %% Filter data frame.
def filterDataFrame(dataFrame, cutoff_frequency=6, order=4):
    
    fs = np.round(1/np.mean(np.diff(dataFrame['time'])), 6)
    fc = cutoff_frequency
    w = fc / (fs / 2)
    if w>=0.999:
        if fc != fs/2:
            print('You tried to filter {}Hz signal with cutoff freq of {}Hz, which is above the Nyquist Frequency. Will filter at {}Hz instead.'.format(fs, fc, fs/2))
        w=0.999
    b, a = signal.butter(order/2, w, 'low')  
    output = signal.filtfilt(
        b, a, dataFrame.loc[:, dataFrame.columns != 'time'], axis=0, 
        padtype='odd', padlen=3*(max(len(b),len(a))-1))
    columns_keys = [i for i in dataFrame.columns if i != 'time']
    output = pd.DataFrame(data=output, columns=columns_keys)
    dataFrameFilt = pd.concat(
        [pd.DataFrame(data=dataFrame['time'].to_numpy(), columns=['time']), 
         output], axis=1)    
    # print('dataFrame filtered at {}Hz.'.format(cutoff_frequency))    
    
    return dataFrameFilt

# %% Extract inverse kinematics data.
def getIK(storage_file, joints, degrees=False):
    
    # Check if data is in degrees or in radians
    table = opensim.TimeSeriesTable(storage_file)
    inDegrees = table.getTableMetaDataString('inDegrees')    
    
    data = storage_to_numpy(storage_file)
    Qs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, joint in enumerate(joints):  
        if ((joint == 'pelvis_tx') or (joint == 'pelvis_ty') or 
            (joint == 'pelvis_tz')):
            Qs.insert(count + 1, joint, data[joint])         
        else:                
            if inDegrees == 'no' and degrees == True:
                Qs.insert(count + 1, joint, data[joint] / np.pi * 180)                
            elif inDegrees == 'yes' and degrees == False:
                Qs.insert(count + 1, joint, data[joint] * np.pi / 180)                
            else:
                Qs.insert(count + 1, joint, data[joint])
                
    return Qs

# %% Get moment arm indices.
def getMomentArmIndices(rightMuscles, leftPolynomialJoints,
                        rightPolynomialJoints, polynomialData):
         
    momentArmIndices = {}
    for count, muscle in enumerate(rightMuscles):        
        spanning = polynomialData[muscle]['spanning']
        for i in range(len(spanning)):
            if (spanning[i] == 1):
                momentArmIndices.setdefault(
                        leftPolynomialJoints[i], []).append(count)
    for count, muscle in enumerate(rightMuscles):        
        spanning = polynomialData[muscle]['spanning']
        for i in range(len(spanning)):
            if (spanning[i] == 1):
                momentArmIndices.setdefault(
                        rightPolynomialJoints[i], []).append(
                                count + len(rightMuscles))                
        
    return momentArmIndices

# %% Get indices in list.
def getIndices(mylist, items):
    
    indices = [mylist.index(item) for item in items]
    
    return indices

# %% Generate external function.
def generateExternalFunction(
        baseDir, dataDir, subject, 
        OpenSimModel="LaiUhlrich2022",
        treadmill=False, build_externalFunction=True, verifyID=True, 
        externalFunctionName='F', overwrite=False,
        useExpressionGraphFunction=True, contact_side='all',trial_name=None):

    # %% Process settings.
    pathCWD = os.getcwd()
    osDir = os.path.join(dataDir, subject, 'OpenSimData')
    pathModelFolder = os.path.join(osDir, 'Model')
    
    # If trial_name is provided, check for trial-specific model folder
    if trial_name is not None:
        trial_model_folder = os.path.join(pathModelFolder, trial_name)
        if os.path.exists(trial_model_folder): # only if is a mono session
            pathModelFolder = trial_model_folder
    
    suffix_MA = '_adjusted'
    if contact_side != 'all':
        suffix_model = '_contacts_' + contact_side
    else:
        suffix_model = '_contacts'
    outputModelFileName = (OpenSimModel + "_scaled" + suffix_MA + suffix_model)
    pathModel = os.path.join(pathModelFolder, outputModelFileName + ".osim")
    pathOutputExternalFunctionFolder = os.path.join(pathModelFolder,
                                                    "ExternalFunction")
    os.makedirs(pathOutputExternalFunctionFolder, exist_ok=True)
    if treadmill:
        externalFunctionName += '_treadmill'
    if contact_side != 'all':
        externalFunctionName += '_' + contact_side
    pathOutputFile = os.path.join(pathOutputExternalFunctionFolder, 
                                  externalFunctionName + ".cpp")
    pathOutputMap = os.path.join(pathOutputExternalFunctionFolder, 
                                 externalFunctionName + "_map.npy")
    
    if useExpressionGraphFunction:
        pathExternalFunction = os.path.join(pathOutputExternalFunctionFolder, 
                                        externalFunctionName + '.py')
    else:
        # This will be deprecated in the future.
        if platform.system() == 'Windows':
            ext_F = '.dll'
        elif platform.system() == 'Darwin':
            ext_F = '.dylib'
        elif platform.system() == 'Linux':
            ext_F = '.so'
        else:
            raise ValueError("Platform not supported.")
        pathExternalFunction = os.path.join(pathOutputExternalFunctionFolder, 
                                    externalFunctionName + ext_F)    
    
    if (overwrite is False and os.path.exists(pathOutputFile) and 
        os.path.exists(pathOutputMap) and os.path.exists(pathExternalFunction)):
        return      
    else:
        print('Generate external function to leverage automatic differentiation.')
    
    # %% Generate external Function (.cpp file)
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathModel)
    model.initSystem()
    bodySet = model.getBodySet()
    jointSet = model.get_JointSet()
    nJoints = jointSet.getSize()
    geometrySet = model.get_ContactGeometrySet()
    forceSet = model.get_ForceSet()
    coordinateSet = model.getCoordinateSet()
    nCoordinates = coordinateSet.getSize()
    coordinates = []
    for coor in range(nCoordinates):
        coordinates.append(coordinateSet.get(coor).getName())
    sides = ['r', 'l']
    for side in sides:
        # We do not include the coordinates from the patellofemoral joints,
        # since they only influence muscle paths, which we approximate using
        # polynomials.
        if 'knee_angle_{}_beta'.format(side) in coordinates:
            nCoordinates -= 1
            nJoints -= 1
    
    nBodies = 0
    for i in range(bodySet.getSize()):        
        c_body = bodySet.get(i)
        c_body_name = c_body.getName()  
        if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
            continue
        nBodies += 1
    
    nContacts = 0
    for i in range(forceSet.getSize()):        
        c_force_elt = forceSet.get(i)        
        if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":  
            nContacts += 1
    
    with open(pathOutputFile, "w") as f:        
        # TODO: only include those that are necessary (model-specific).
        f.write('#include <OpenSim/Simulation/Model/Model.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/PinJoint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/WeldJoint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/Joint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/SpatialTransform.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/CustomJoint.h>\n')
        if treadmill:
            f.write('#include <OpenSim/Simulation/SimbodyEngine/SliderJoint.h>\n')    
        f.write('#include <OpenSim/Common/LinearFunction.h>\n')
        f.write('#include <OpenSim/Common/PolynomialFunction.h>\n')
        f.write('#include <OpenSim/Common/MultiplierFunction.h>\n')
        f.write('#include <OpenSim/Common/Constant.h>\n')
        f.write('#include <OpenSim/Simulation/Model/SmoothSphereHalfSpaceForce.h>\n')
        f.write('#include <OpenSim/Simulation/SimulationUtilities.h>\n')
        f.write('#include "SimTKcommon/internal/recorder.h"\n\n')
        
        f.write('#include <iostream>\n')
        f.write('#include <iterator>\n')
        f.write('#include <random>\n')
        f.write('#include <cassert>\n')
        f.write('#include <algorithm>\n')
        f.write('#include <vector>\n')
        f.write('#include <fstream>\n\n')
        
        f.write('using namespace SimTK;\n')
        f.write('using namespace OpenSim;\n\n')
    
        if treadmill:
            f.write('constexpr int n_in = 3; \n')
        else:
            f.write('constexpr int n_in = 2; \n')
        f.write('constexpr int n_out = 1; \n')
        
        f.write('constexpr int nCoordinates = %i; \n' % nCoordinates)
        f.write('constexpr int NX = nCoordinates*2; \n')
        f.write('constexpr int NU = nCoordinates; \n\n')
        if treadmill:
            nCoordinates_treadmill = nCoordinates + 1
            f.write('constexpr int nCoordinates_treadmill = %i; \n' % nCoordinates_treadmill)
            f.write('constexpr int NX_treadmill = nCoordinates_treadmill*2; \n')
            f.write('constexpr int NU_treadmill = nCoordinates_treadmill; \n\n')
    
        f.write('template<typename T> \n')
        f.write('T value(const Recorder& e) { return e; }; \n')
        f.write('template<> \n')
        f.write('double value(const Recorder& e) { return e.getValue(); }; \n\n')
        
        f.write('template<typename T>\n')
        f.write('int F_generic(const T** arg, T** res) {\n\n')
        
        # Model
        f.write('\t// Definition of model.\n')
        f.write('\tOpenSim::Model* model;\n')
        f.write('\tmodel = new OpenSim::Model();\n\n')
        
        # Bodies
        f.write('\t// Definition of bodies.\n')
        for i in range(bodySet.getSize()):        
            c_body = bodySet.get(i)
            c_body_name = c_body.getName()            
            if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
                continue            
            c_body_mass = c_body.get_mass()
            c_body_mass_center = c_body.get_mass_center().to_numpy()
            c_body_inertia = c_body.get_inertia()
            c_body_inertia_vec3 = np.array([c_body_inertia.get(0), c_body_inertia.get(1), c_body_inertia.get(2)])        
            f.write('\tOpenSim::Body* %s;\n' % c_body_name)
            f.write('\t%s = new OpenSim::Body(\"%s\", %.20f, Vec3(%.20f, %.20f, %.20f), Inertia(%.20f, %.20f, %.20f, 0., 0., 0.));\n' % (c_body_name, c_body_name, c_body_mass, c_body_mass_center[0], c_body_mass_center[1], c_body_mass_center[2], c_body_inertia_vec3[0], c_body_inertia_vec3[1], c_body_inertia_vec3[2]))
            f.write('\tmodel->addBody(%s);\n' % (c_body_name))
            f.write('\n')
        if treadmill:
            f.write('\tOpenSim::Body* treadmill;\n')
            f.write('\ttreadmill = new OpenSim::Body("treadmill", 1., Vec3(0), Inertia(1,1,1,0,0,0));\n')
            f.write('\tmodel->addBody(treadmill);\n')
            f.write('\n')
        
        # Joints
        f.write('\t// Definition of joints.\n')
        for i in range(jointSet.getSize()): 
            c_joint = jointSet.get(i)
            c_joint_type = c_joint.getConcreteClassName()
            
            c_joint_name = c_joint.getName()
            if (c_joint_name == 'patellofemoral_l' or 
                c_joint_name == 'patellofemoral_r'):
                continue
            
            parent_frame = c_joint.get_frames(0)
            parent_frame_name = parent_frame.getParentFrame().getName()
            parent_frame_trans = parent_frame.get_translation().to_numpy()
            parent_frame_or = parent_frame.get_orientation().to_numpy()
            
            child_frame = c_joint.get_frames(1)
            child_frame_name = child_frame.getParentFrame().getName()
            child_frame_trans = child_frame.get_translation().to_numpy()
            child_frame_or = child_frame.get_orientation().to_numpy()
            
            # Custom joints
            if c_joint_type == "CustomJoint":
                
                f.write('\tSpatialTransform st_%s;\n' % c_joint.getName())                
                cObj = opensim.CustomJoint.safeDownCast(c_joint)    
                spatialtransform = cObj.get_SpatialTransform()
                
                # Transform axis.
                # Rotation 1
                rot1 = spatialtransform.get_rotation1()
                rot1_axis = rot1.get_axis().to_numpy()
                rot1_f = rot1.get_function()
                coord = 0
                if rot1_f.getConcreteClassName() == 'LinearFunction':  
                    rot1_f_obj = opensim.LinearFunction.safeDownCast(rot1_f)                          
                    rot1_f_slope = rot1_f_obj.getSlope()
                    rot1_f_intercept = rot1_f_obj.getIntercept()                
                    c_coord_name = rot1.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    f.write('\tst_%s[%i].setFunction(new LinearFunction(%.4f, %.4f));\n' % (c_joint.getName(), coord, rot1_f_slope, rot1_f_intercept))                
                elif rot1_f.getConcreteClassName() == 'PolynomialFunction':
                    c_coord_name = rot1.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    rot1_f_obj = opensim.PolynomialFunction.safeDownCast(rot1_f)                
                    rot1_f_coeffs = rot1_f_obj.getCoefficients().to_numpy()
                    c_nCoeffs = rot1_f_coeffs.shape[0]                
                    if c_nCoeffs == 2:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot1_f_coeffs[0], rot1_f_coeffs[1]))
                    elif c_nCoeffs == 3:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot1_f_coeffs[0], rot1_f_coeffs[1], rot1_f_coeffs[2]))
                    elif c_nCoeffs == 4:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot1_f_coeffs[0], rot1_f_coeffs[1], rot1_f_coeffs[2], rot1_f_coeffs[3]))  
                    elif c_nCoeffs == 5:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot1_f_coeffs[0], rot1_f_coeffs[1], rot1_f_coeffs[2], rot1_f_coeffs[3], rot1_f_coeffs[4]))                    
                    else:
                        raise ValueError("TODO")
                    f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                    f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                    f.write('\tst_%s[%i].setFunction(new PolynomialFunction(st_%s_%i_coeffs_vec));\n' % (c_joint.getName(), coord, c_joint.getName(), coord))
                elif rot1_f.getConcreteClassName() == 'MultiplierFunction':
                    rot1_f_obj = opensim.MultiplierFunction.safeDownCast(rot1_f)
                    rot1_f_obj_scale = rot1_f_obj.getScale()
                    rot1_f_obj_f = rot1_f_obj.getFunction()
                    rot1_f_obj_f_name = rot1_f_obj_f.getConcreteClassName()
                    if rot1_f_obj_f_name == 'Constant':
                        rot1_f_obj_f_obj = opensim.Constant.safeDownCast(rot1_f_obj_f)
                        rot1_f_obj_f_obj_value = rot1_f_obj_f_obj.getValue()
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new Constant(%.20f), %.20f));\n' % (c_joint.getName(), coord, rot1_f_obj_f_obj_value, rot1_f_obj_scale))
                    elif rot1_f_obj_f_name == 'PolynomialFunction':
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                        rot1_f_obj_f_obj = opensim.PolynomialFunction.safeDownCast(rot1_f_obj_f)
                        rot1_f_obj_f_coeffs = rot1_f_obj_f_obj.getCoefficients().to_numpy()
                        c_nCoeffs = rot1_f_obj_f_coeffs.shape[0]
                        if c_nCoeffs == 2:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot1_f_obj_f_coeffs[0], rot1_f_obj_f_coeffs[1]))
                        elif c_nCoeffs == 3:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot1_f_obj_f_coeffs[0], rot1_f_obj_f_coeffs[1], rot1_f_obj_f_coeffs[2]))
                        elif c_nCoeffs == 4:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot1_f_obj_f_coeffs[0], rot1_f_obj_f_coeffs[1], rot1_f_obj_f_coeffs[2], rot1_f_obj_f_coeffs[3]))  
                        elif c_nCoeffs == 5:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot1_f_obj_f_coeffs[0], rot1_f_obj_f_coeffs[1], rot1_f_obj_f_coeffs[2], rot1_f_obj_f_coeffs[3], rot1_f_obj_f_coeffs[4]))                    
                        else:
                            raise ValueError("TODO")
                        f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                        f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new PolynomialFunction(st_%s_%i_coeffs_vec), %.20f));\n' % (c_joint.getName(), coord, c_joint.getName(), coord, rot1_f_obj_scale))
                    else:
                        raise ValueError("Not supported")
                elif rot1_f.getConcreteClassName() == 'Constant':
                    rot1_f_obj = opensim.Constant.safeDownCast(rot1_f)
                    rot1_f_obj_value = rot1_f_obj.getValue()
                    f.write('\tst_%s[%i].setFunction(new Constant(%.20f));\n' % (c_joint.getName(), coord, rot1_f_obj_value))
                else:
                    raise ValueError("Not supported")
                f.write('\tst_%s[%i].setAxis(Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), coord, rot1_axis[0], rot1_axis[1], rot1_axis[2]))
                
                # Rotation 2
                rot2 = spatialtransform.get_rotation2()
                rot2_axis = rot2.get_axis().to_numpy()
                rot2_f = rot2.get_function()
                coord = 1
                if rot2_f.getConcreteClassName() == 'LinearFunction':
                    rot2_f_obj = opensim.LinearFunction.safeDownCast(rot2_f)
                    rot2_f_slope = rot2_f_obj.getSlope()
                    rot2_f_intercept = rot2_f_obj.getIntercept()                
                    c_coord_name = rot2.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    f.write('\tst_%s[%i].setFunction(new LinearFunction(%.4f, %.4f));\n' % (c_joint.getName(), coord, rot2_f_slope, rot2_f_intercept))
                elif rot2_f.getConcreteClassName() == 'PolynomialFunction':
                    c_coord_name = rot2.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    rot2_f_obj = opensim.PolynomialFunction.safeDownCast(rot2_f)                
                    rot2_f_coeffs = rot2_f_obj.getCoefficients().to_numpy()
                    c_nCoeffs = rot2_f_coeffs.shape[0]                
                    if c_nCoeffs == 2:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot2_f_coeffs[0], rot2_f_coeffs[1]))
                    elif c_nCoeffs == 3:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot2_f_coeffs[0], rot2_f_coeffs[1], rot2_f_coeffs[2]))
                    elif c_nCoeffs == 4:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot2_f_coeffs[0], rot2_f_coeffs[1], rot2_f_coeffs[2], rot2_f_coeffs[3]))  
                    elif c_nCoeffs == 5:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot2_f_coeffs[0], rot2_f_coeffs[1], rot2_f_coeffs[2], rot2_f_coeffs[3], rot2_f_coeffs[4]))                    
                    else:
                        raise ValueError("TODO")
                    f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                    f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                    f.write('\tst_%s[%i].setFunction(new PolynomialFunction(st_%s_%i_coeffs_vec));\n' % (c_joint.getName(), coord, c_joint.getName(), coord))
                elif rot2_f.getConcreteClassName() == 'MultiplierFunction':
                    rot2_f_obj = opensim.MultiplierFunction.safeDownCast(rot2_f)
                    rot2_f_obj_scale = rot2_f_obj.getScale()
                    rot2_f_obj_f = rot2_f_obj.getFunction()
                    rot2_f_obj_f_name = rot2_f_obj_f.getConcreteClassName()
                    if rot2_f_obj_f_name == 'Constant':
                        rot2_f_obj_f_obj = opensim.Constant.safeDownCast(rot2_f_obj_f)
                        rot2_f_obj_f_obj_value = rot2_f_obj_f_obj.getValue()
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new Constant(%.20f), %.20f));\n' % (c_joint.getName(), coord, rot2_f_obj_f_obj_value, rot2_f_obj_scale)) 
                    elif rot2_f_obj_f_name == 'PolynomialFunction':
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                        rot2_f_obj_f_obj = opensim.PolynomialFunction.safeDownCast(rot2_f_obj_f)
                        rot2_f_obj_f_coeffs = rot2_f_obj_f_obj.getCoefficients().to_numpy()
                        c_nCoeffs = rot2_f_obj_f_coeffs.shape[0]
                        if c_nCoeffs == 2:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot2_f_obj_f_coeffs[0], rot2_f_obj_f_coeffs[1]))
                        elif c_nCoeffs == 3:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot2_f_obj_f_coeffs[0], rot2_f_obj_f_coeffs[1], rot2_f_obj_f_coeffs[2]))
                        elif c_nCoeffs == 4:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot2_f_obj_f_coeffs[0], rot2_f_obj_f_coeffs[1], rot2_f_obj_f_coeffs[2], rot2_f_obj_f_coeffs[3]))  
                        elif c_nCoeffs == 5:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot2_f_obj_f_coeffs[0], rot2_f_obj_f_coeffs[1], rot2_f_obj_f_coeffs[2], rot2_f_obj_f_coeffs[3], rot2_f_obj_f_coeffs[4]))                    
                        else:
                            raise ValueError("TODO")
                        f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                        f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new PolynomialFunction(st_%s_%i_coeffs_vec), %.20f));\n' % (c_joint.getName(), coord, c_joint.getName(), coord, rot2_f_obj_scale))
                    else:
                        raise ValueError("Not supported")
                elif rot2_f.getConcreteClassName() == 'Constant':
                    rot2_f_obj = opensim.Constant.safeDownCast(rot2_f)
                    rot2_f_obj_value = rot2_f_obj.getValue()
                    f.write('\tst_%s[%i].setFunction(new Constant(%.20f));\n' % (c_joint.getName(), coord, rot2_f_obj_value))
                else:
                    raise ValueError("Not supported")
                f.write('\tst_%s[%i].setAxis(Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), coord, rot2_axis[0], rot2_axis[1], rot2_axis[2]))
                
                # Rotation 3
                rot3 = spatialtransform.get_rotation3()
                rot3_axis = rot3.get_axis().to_numpy()
                rot3_f = rot3.get_function()
                coord = 2
                if rot3_f.getConcreteClassName() == 'LinearFunction': 
                    rot3_f_obj = opensim.LinearFunction.safeDownCast(rot3_f)
                    rot3_f_slope = rot3_f_obj.getSlope()
                    rot3_f_intercept = rot3_f_obj.getIntercept()                
                    c_coord_name = rot3.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    f.write('\tst_%s[%i].setFunction(new LinearFunction(%.4f, %.4f));\n' % (c_joint.getName(), coord, rot3_f_slope, rot3_f_intercept))
                elif rot3_f.getConcreteClassName() == 'PolynomialFunction':
                    c_coord_name = rot3.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    rot3_f_obj = opensim.PolynomialFunction.safeDownCast(rot3_f)                
                    rot3_f_coeffs = rot3_f_obj.getCoefficients().to_numpy()
                    c_nCoeffs = rot3_f_coeffs.shape[0]                
                    if c_nCoeffs == 2:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot3_f_coeffs[0], rot3_f_coeffs[1]))
                    elif c_nCoeffs == 3:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot3_f_coeffs[0], rot3_f_coeffs[1], rot3_f_coeffs[2]))
                    elif c_nCoeffs == 4:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot3_f_coeffs[0], rot3_f_coeffs[1], rot3_f_coeffs[2], rot3_f_coeffs[3]))  
                    elif c_nCoeffs == 5:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot3_f_coeffs[0], rot3_f_coeffs[1], rot3_f_coeffs[2], rot3_f_coeffs[3], rot3_f_coeffs[4]))                    
                    else:
                        raise ValueError("TODO")
                    f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                    f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                    f.write('\tst_%s[%i].setFunction(new PolynomialFunction(st_%s_%i_coeffs_vec));\n' % (c_joint.getName(), coord, c_joint.getName(), coord))
                elif rot3_f.getConcreteClassName() == 'MultiplierFunction':
                    rot3_f_obj = opensim.MultiplierFunction.safeDownCast(rot3_f)
                    rot3_f_obj_scale = rot3_f_obj.getScale()
                    rot3_f_obj_f = rot3_f_obj.getFunction()
                    rot3_f_obj_f_name = rot3_f_obj_f.getConcreteClassName()
                    if rot3_f_obj_f_name == 'Constant':
                        rot3_f_obj_f_obj = opensim.Constant.safeDownCast(rot3_f_obj_f)
                        rot3_f_obj_f_obj_value = rot3_f_obj_f_obj.getValue()
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new Constant(%.20f), %.20f));\n' % (c_joint.getName(), coord, rot3_f_obj_f_obj_value, rot3_f_obj_scale))
                    elif rot3_f_obj_f_name == 'PolynomialFunction':
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                        rot3_f_obj_f_obj = opensim.PolynomialFunction.safeDownCast(rot3_f_obj_f)
                        rot3_f_obj_f_coeffs = rot3_f_obj_f_obj.getCoefficients().to_numpy()
                        c_nCoeffs = rot3_f_obj_f_coeffs.shape[0]
                        if c_nCoeffs == 2:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot3_f_obj_f_coeffs[0], rot3_f_obj_f_coeffs[1]))
                        elif c_nCoeffs == 3:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot3_f_obj_f_coeffs[0], rot3_f_obj_f_coeffs[1], rot3_f_obj_f_coeffs[2]))
                        elif c_nCoeffs == 4:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot3_f_obj_f_coeffs[0], rot3_f_obj_f_coeffs[1], rot3_f_obj_f_coeffs[2], rot3_f_obj_f_coeffs[3]))  
                        elif c_nCoeffs == 5:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, rot3_f_obj_f_coeffs[0], rot3_f_obj_f_coeffs[1], rot3_f_obj_f_coeffs[2], rot3_f_obj_f_coeffs[3], rot3_f_obj_f_coeffs[4]))                    
                        else:
                            raise ValueError("TODO")
                        f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                        f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new PolynomialFunction(st_%s_%i_coeffs_vec), %.20f));\n' % (c_joint.getName(), coord, c_joint.getName(), coord, rot3_f_obj_scale))
                    else:
                        raise ValueError("Not supported")
                elif rot3_f.getConcreteClassName() == 'Constant':
                    rot3_f_obj = opensim.Constant.safeDownCast(rot3_f)
                    rot3_f_obj_value = rot3_f_obj.getValue()
                    f.write('\tst_%s[%i].setFunction(new Constant(%.20f));\n' % (c_joint.getName(), coord, rot3_f_obj_value))
                else:
                    raise ValueError("Not supported")
                f.write('\tst_%s[%i].setAxis(Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), coord, rot3_axis[0], rot3_axis[1], rot3_axis[2]))
                
                # Translation 1
                tr1 = spatialtransform.get_translation1()
                tr1_axis = tr1.get_axis().to_numpy()
                tr1_f = tr1.get_function()
                coord = 3
                if tr1_f.getConcreteClassName() == 'LinearFunction':    
                    tr1_f_obj = opensim.LinearFunction.safeDownCast(tr1_f)
                    tr1_f_slope = tr1_f_obj.getSlope()
                    tr1_f_intercept = tr1_f_obj.getIntercept()
                    c_coord_name = tr1.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    f.write('\tst_%s[%i].setFunction(new LinearFunction(%.4f, %.4f));\n' % (c_joint.getName(), coord, tr1_f_slope, tr1_f_intercept))
                elif tr1_f.getConcreteClassName() == 'PolynomialFunction':
                    c_coord_name = tr1.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    tr1_f_obj = opensim.PolynomialFunction.safeDownCast(tr1_f)                
                    tr1_f_coeffs = tr1_f_obj.getCoefficients().to_numpy()
                    c_nCoeffs = tr1_f_coeffs.shape[0]
                    if c_nCoeffs == 2:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr1_f_coeffs[0], tr1_f_coeffs[1]))
                    elif c_nCoeffs == 3:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr1_f_coeffs[0], tr1_f_coeffs[1], tr1_f_coeffs[2]))
                    elif c_nCoeffs == 4:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr1_f_coeffs[0], tr1_f_coeffs[1], tr1_f_coeffs[2], tr1_f_coeffs[3]))  
                    elif c_nCoeffs == 5:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr1_f_coeffs[0], tr1_f_coeffs[1], tr1_f_coeffs[2], tr1_f_coeffs[3], tr1_f_coeffs[4]))                    
                    else:
                        raise ValueError("TODO")
                    f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                    f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                    f.write('\tst_%s[%i].setFunction(new PolynomialFunction(st_%s_%i_coeffs_vec));\n' % (c_joint.getName(), coord, c_joint.getName(), coord))
                elif tr1_f.getConcreteClassName() == 'MultiplierFunction':
                    tr1_f_obj = opensim.MultiplierFunction.safeDownCast(tr1_f)
                    tr1_f_obj_scale = tr1_f_obj.getScale()
                    tr1_f_obj_f = tr1_f_obj.getFunction()
                    tr1_f_obj_f_name = tr1_f_obj_f.getConcreteClassName()
                    if tr1_f_obj_f_name == 'Constant':
                        tr1_f_obj_f_obj = opensim.Constant.safeDownCast(tr1_f_obj_f)
                        tr1_f_obj_f_obj_value = tr1_f_obj_f_obj.getValue()
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new Constant(%.20f), %.20f));\n' % (c_joint.getName(), coord, tr1_f_obj_f_obj_value, tr1_f_obj_scale))
                    elif tr1_f_obj_f_name == 'PolynomialFunction':
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                        tr1_f_obj_f_obj = opensim.PolynomialFunction.safeDownCast(tr1_f_obj_f)
                        tr1_f_obj_f_coeffs = tr1_f_obj_f_obj.getCoefficients().to_numpy()
                        c_nCoeffs = tr1_f_obj_f_coeffs.shape[0]
                        if c_nCoeffs == 2:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr1_f_obj_f_coeffs[0], tr1_f_obj_f_coeffs[1]))
                        elif c_nCoeffs == 3:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr1_f_obj_f_coeffs[0], tr1_f_obj_f_coeffs[1], tr1_f_obj_f_coeffs[2]))
                        elif c_nCoeffs == 4:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr1_f_obj_f_coeffs[0], tr1_f_obj_f_coeffs[1], tr1_f_obj_f_coeffs[2], tr1_f_obj_f_coeffs[3]))  
                        elif c_nCoeffs == 5:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr1_f_obj_f_coeffs[0], tr1_f_obj_f_coeffs[1], tr1_f_obj_f_coeffs[2], tr1_f_obj_f_coeffs[3], tr1_f_obj_f_coeffs[4]))                    
                        else:
                            raise ValueError("TODO")
                        f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                        f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new PolynomialFunction(st_%s_%i_coeffs_vec), %.20f));\n' % (c_joint.getName(), coord, c_joint.getName(), coord, tr1_f_obj_scale))
                    else:
                        raise ValueError("Not supported")
                elif tr1_f.getConcreteClassName() == 'Constant':
                    tr1_f_obj = opensim.Constant.safeDownCast(tr1_f)
                    tr1_f_obj_value = tr1_f_obj.getValue()
                    f.write('\tst_%s[%i].setFunction(new Constant(%.20f));\n' % (c_joint.getName(), coord, tr1_f_obj_value))
                else:
                    raise ValueError("Not supported")
                f.write('\tst_%s[%i].setAxis(Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), coord, tr1_axis[0], tr1_axis[1], tr1_axis[2]))            
                
                # Translation 2
                tr2 = spatialtransform.get_translation2()
                tr2_axis = tr2.get_axis().to_numpy()
                tr2_f = tr2.get_function()
                coord = 4
                if tr2_f.getConcreteClassName() == 'LinearFunction': 
                    tr2_f_obj = opensim.LinearFunction.safeDownCast(tr2_f)
                    tr2_f_slope = tr2_f_obj.getSlope()
                    tr2_f_intercept = tr2_f_obj.getIntercept()                
                    c_coord_name = tr2.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    f.write('\tst_%s[%i].setFunction(new LinearFunction(%.4f, %.4f));\n' % (c_joint.getName(), coord, tr2_f_slope, tr2_f_intercept))
                elif tr2_f.getConcreteClassName() == 'PolynomialFunction':
                    c_coord_name = tr2.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    tr2_f_obj = opensim.PolynomialFunction.safeDownCast(tr2_f)                
                    tr2_f_coeffs = tr2_f_obj.getCoefficients().to_numpy()
                    c_nCoeffs = tr2_f_coeffs.shape[0]                
                    if c_nCoeffs == 2:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr2_f_coeffs[0], tr2_f_coeffs[1]))
                    elif c_nCoeffs == 3:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr2_f_coeffs[0], tr2_f_coeffs[1], tr2_f_coeffs[2]))
                    elif c_nCoeffs == 4:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr2_f_coeffs[0], tr2_f_coeffs[1], tr2_f_coeffs[2], tr2_f_coeffs[3]))  
                    elif c_nCoeffs == 5:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr2_f_coeffs[0], tr2_f_coeffs[1], tr2_f_coeffs[2], tr2_f_coeffs[3], tr2_f_coeffs[4]))                    
                    else:
                        raise ValueError("TODO")
                    f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                    f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                    f.write('\tst_%s[%i].setFunction(new PolynomialFunction(st_%s_%i_coeffs_vec));\n' % (c_joint.getName(), coord, c_joint.getName(), coord))
                elif tr2_f.getConcreteClassName() == 'MultiplierFunction':
                    tr2_f_obj = opensim.MultiplierFunction.safeDownCast(tr2_f)
                    tr2_f_obj_scale = tr2_f_obj.getScale()
                    tr2_f_obj_f = tr2_f_obj.getFunction()
                    tr2_f_obj_f_name = tr2_f_obj_f.getConcreteClassName()
                    if tr2_f_obj_f_name == 'Constant':
                        tr2_f_obj_f_obj = opensim.Constant.safeDownCast(tr2_f_obj_f)
                        tr2_f_obj_f_obj_value = tr2_f_obj_f_obj.getValue()
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new Constant(%.20f), %.20f));\n' % (c_joint.getName(), coord, tr2_f_obj_f_obj_value, tr2_f_obj_scale))
                    elif tr2_f_obj_f_name == 'PolynomialFunction':
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                        tr2_f_obj_f_obj = opensim.PolynomialFunction.safeDownCast(tr2_f_obj_f)
                        tr2_f_obj_f_coeffs = tr2_f_obj_f_obj.getCoefficients().to_numpy()
                        c_nCoeffs = tr2_f_obj_f_coeffs.shape[0]
                        if c_nCoeffs == 2:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr2_f_obj_f_coeffs[0], tr2_f_obj_f_coeffs[1]))
                        elif c_nCoeffs == 3:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr2_f_obj_f_coeffs[0], tr2_f_obj_f_coeffs[1], tr2_f_obj_f_coeffs[2]))
                        elif c_nCoeffs == 4:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr2_f_obj_f_coeffs[0], tr2_f_obj_f_coeffs[1], tr2_f_obj_f_coeffs[2], tr2_f_obj_f_coeffs[3]))  
                        elif c_nCoeffs == 5:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr2_f_obj_f_coeffs[0], tr2_f_obj_f_coeffs[1], tr2_f_obj_f_coeffs[2], tr2_f_obj_f_coeffs[3], tr2_f_obj_f_coeffs[4]))                    
                        else:
                            raise ValueError("TODO")
                        f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                        f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new PolynomialFunction(st_%s_%i_coeffs_vec), %.20f));\n' % (c_joint.getName(), coord, c_joint.getName(), coord, tr2_f_obj_scale))
                    else:
                        raise ValueError("Not supported")
                elif tr2_f.getConcreteClassName() == 'Constant':
                    tr2_f_obj = opensim.Constant.safeDownCast(tr2_f)
                    tr2_f_obj_value = tr2_f_obj.getValue()
                    f.write('\tst_%s[%i].setFunction(new Constant(%.20f));\n' % (c_joint.getName(), coord, tr2_f_obj_value))
                else:
                    raise ValueError("Not supported")
                f.write('\tst_%s[%i].setAxis(Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), coord, tr2_axis[0], tr2_axis[1], tr2_axis[2]))
                
                # Translation 3
                tr3 = spatialtransform.get_translation3()
                tr3_axis = tr3.get_axis().to_numpy()
                tr3_f = tr3.get_function()
                coord = 5
                if tr3_f.getConcreteClassName() == 'LinearFunction':     
                    tr3_f_obj = opensim.LinearFunction.safeDownCast(tr3_f)
                    tr3_f_slope = tr3_f_obj.getSlope()
                    tr3_f_intercept = tr3_f_obj.getIntercept()                
                    c_coord_name = tr3.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    f.write('\tst_%s[%i].setFunction(new LinearFunction(%.4f, %.4f));\n' % (c_joint.getName(), coord, tr3_f_slope, tr3_f_intercept))
                elif tr3_f.getConcreteClassName() == 'PolynomialFunction':
                    c_coord_name = tr3.get_coordinates(0)
                    f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                    tr3_f_obj = opensim.PolynomialFunction.safeDownCast(tr3_f)                
                    tr3_f_coeffs = tr3_f_obj.getCoefficients().to_numpy()
                    c_nCoeffs = tr3_f_coeffs.shape[0]                
                    if c_nCoeffs == 2:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr3_f_coeffs[0], tr3_f_coeffs[1]))
                    elif c_nCoeffs == 3:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr3_f_coeffs[0], tr3_f_coeffs[1], tr3_f_coeffs[2]))
                    elif c_nCoeffs == 4:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr3_f_coeffs[0], tr3_f_coeffs[1], tr3_f_coeffs[2], tr3_f_coeffs[3]))  
                    elif c_nCoeffs == 5:
                        f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr3_f_coeffs[0], tr3_f_coeffs[1], tr3_f_coeffs[2], tr3_f_coeffs[3], tr3_f_coeffs[4]))                    
                    else:
                        raise ValueError("TODO")
                    f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                    f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                    f.write('\tst_%s[%i].setFunction(new PolynomialFunction(st_%s_%i_coeffs_vec));\n' % (c_joint.getName(), coord, c_joint.getName(), coord))
                elif tr3_f.getConcreteClassName() == 'MultiplierFunction':
                    tr3_f_obj = opensim.MultiplierFunction.safeDownCast(tr3_f)
                    tr3_f_obj_scale = tr3_f_obj.getScale()
                    tr3_f_obj_f = tr3_f_obj.getFunction()
                    tr3_f_obj_f_name = tr3_f_obj_f.getConcreteClassName()
                    if tr3_f_obj_f_name == 'Constant':
                        tr3_f_obj_f_obj = opensim.Constant.safeDownCast(tr3_f_obj_f)
                        tr3_f_obj_f_obj_value = tr3_f_obj_f_obj.getValue()
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new Constant(%.20f), %.20f));\n' % (c_joint.getName(), coord, tr3_f_obj_f_obj_value, tr3_f_obj_scale))
                    elif tr3_f_obj_f_name == 'PolynomialFunction':
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (c_joint.getName(), coord, c_coord_name))
                        tr3_f_obj_f_obj = opensim.PolynomialFunction.safeDownCast(tr3_f_obj_f)
                        tr3_f_obj_f_coeffs = tr3_f_obj_f_obj.getCoefficients().to_numpy()
                        c_nCoeffs = tr3_f_obj_f_coeffs.shape[0]
                        if c_nCoeffs == 2:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr3_f_obj_f_coeffs[0], tr3_f_obj_f_coeffs[1]))
                        elif c_nCoeffs == 3:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr3_f_obj_f_coeffs[0], tr3_f_obj_f_coeffs[1], tr3_f_obj_f_coeffs[2]))
                        elif c_nCoeffs == 4:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr3_f_obj_f_coeffs[0], tr3_f_obj_f_coeffs[1], tr3_f_obj_f_coeffs[2], tr3_f_obj_f_coeffs[3]))  
                        elif c_nCoeffs == 5:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (c_joint.getName(), coord, c_nCoeffs, tr3_f_obj_f_coeffs[0], tr3_f_obj_f_coeffs[1], tr3_f_obj_f_coeffs[2], tr3_f_obj_f_coeffs[3], tr3_f_obj_f_coeffs[4]))                    
                        else:
                            raise ValueError("TODO")
                        f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                        f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                        f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new PolynomialFunction(st_%s_%i_coeffs_vec), %.20f));\n' % (c_joint.getName(), coord, c_joint.getName(), coord, tr3_f_obj_scale))
                    else:
                        raise ValueError("Not supported") 
                elif tr3_f.getConcreteClassName() == 'Constant':
                    tr3_f_obj = opensim.Constant.safeDownCast(tr3_f)
                    tr3_f_obj_value = tr3_f_obj.getValue()
                    f.write('\tst_%s[%i].setFunction(new Constant(%.20f));\n' % (c_joint.getName(), coord, tr3_f_obj_value))
                else:
                    raise ValueError("Not supported")
                f.write('\tst_%s[%i].setAxis(Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), coord, tr3_axis[0], tr3_axis[1], tr3_axis[2]))          
                
                # Joint.
                f.write('\tOpenSim::%s* %s;\n' % (c_joint_type, c_joint.getName()))
                if parent_frame_name == "ground":
                    f.write('\t%s = new OpenSim::%s(\"%s\", model->getGround(), Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), st_%s);\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2], c_joint.getName()))     
                else:
                    f.write('\t%s = new OpenSim::%s(\"%s\", *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), st_%s);\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_name, parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2], c_joint.getName()))
                
            elif c_joint_type == 'PinJoint' or c_joint_type == 'WeldJoint' :
                f.write('\tOpenSim::%s* %s;\n' % (c_joint_type, c_joint.getName()))
                if parent_frame_name == "ground":
                    f.write('\t%s = new OpenSim::%s(\"%s\", model->getGround(), Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2]))     
                else:
                    f.write('\t%s = new OpenSim::%s(\"%s\", *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_name, parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2])) 
            else:
                raise ValueError("TODO: joint type not yet supported")
            f.write('\tmodel->addJoint(%s);\n' % (c_joint.getName()))
            f.write('\n')  
        if treadmill:
            f.write('\tOpenSim::SliderJoint* ground_treadmill;\n')
            f.write('\tground_treadmill = new SliderJoint("ground_treadmill", model->getGround(), Vec3(0), Vec3(0), *treadmill, Vec3(0), Vec3(0));\n')
            f.write('\tmodel->addJoint(ground_treadmill);\n')
            f.write('\n')
            
        # Contacts
        f.write('\t// Definition of contacts.\n')
        rightFootContact = False
        leftFootContact = False
        rightFootContactBodies = []
        leftFootContactBodies = []
        nRightContacts = 0
        nLeftContacts = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":            
                c_force_elt_obj =  opensim.SmoothSphereHalfSpaceForce.safeDownCast(c_force_elt) 	
                
                socket0Name = c_force_elt.getSocketNames()[0]
                socket0 = c_force_elt.getSocket(socket0Name)
                socket0_obj = socket0.getConnecteeAsObject()
                socket0_objName = socket0_obj.getName()            
                geo0 = geometrySet.get(socket0_objName)
                geo0_loc = geo0.get_location().to_numpy()
                geo0_or = geo0.get_orientation().to_numpy()
                geo0_frameName = geo0.getFrame().getName()
                
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_loc = geo1.get_location().to_numpy()
                geo1_frameName = geo1.getFrame().getName()
                obj = opensim.ContactSphere.safeDownCast(geo1) 	
                geo1_radius = obj.getRadius()            
                
                f.write('\tOpenSim::%s* %s;\n' % (c_force_elt.getConcreteClassName(), c_force_elt.getName()))
                if geo0_frameName == "ground":
                    if treadmill:
                        ground_contact = "*treadmill"
                    else:
                        ground_contact = "model->getGround()"
                    
                    f.write('\t%s = new %s(\"%s\", *%s, %s);\n' % (c_force_elt.getName(), c_force_elt.getConcreteClassName(), c_force_elt.getName(), geo1_frameName, ground_contact))
                else:
                    f.write('\t%s = new %s(\"%s\", *%s, *%s);\n' % (c_force_elt.getName(), c_force_elt.getConcreteClassName(), c_force_elt.getName(), geo1_frameName, geo0_frameName))
                    
                f.write('\tVec3 %s_location(%.20f, %.20f, %.20f);\n' % (c_force_elt.getName(), geo1_loc[0], geo1_loc[1], geo1_loc[2]))
                f.write('\t%s->set_contact_sphere_location(%s_location);\n' % (c_force_elt.getName(), c_force_elt.getName()))
                f.write('\tdouble %s_radius = (%.20f);\n' % (c_force_elt.getName(), geo1_radius))
                f.write('\t%s->set_contact_sphere_radius(%s_radius );\n' % (c_force_elt.getName(), c_force_elt.getName()))
                f.write('\t%s->set_contact_half_space_location(Vec3(%.20f, %.20f, %.20f));\n' % (c_force_elt.getName(), geo0_loc[0], geo0_loc[1], geo0_loc[2]))
                f.write('\t%s->set_contact_half_space_orientation(Vec3(%.20f, %.20f, %.20f));\n' % (c_force_elt.getName(), geo0_or[0], geo0_or[1], geo0_or[2]))
                
                f.write('\t%s->set_stiffness(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_stiffness()))
                f.write('\t%s->set_dissipation(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_dissipation()))
                f.write('\t%s->set_static_friction(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_static_friction()))
                f.write('\t%s->set_dynamic_friction(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_dynamic_friction()))
                f.write('\t%s->set_viscous_friction(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_viscous_friction()))
                f.write('\t%s->set_transition_velocity(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_transition_velocity()))
                
                f.write('\t%s->connectSocket_sphere_frame(*%s);\n' % (c_force_elt.getName(), geo1_frameName))
                if geo0_frameName == "ground":
                    f.write('\t%s->connectSocket_half_space_frame(%s);\n' % (c_force_elt.getName(), ground_contact))                
                else:
                    f.write('\t%s->connectSocket_half_space_frame(*%s);\n' % (c_force_elt.getName(), geo0_frameName))
                f.write('\tmodel->addComponent(%s);\n' % (c_force_elt.getName()))
                f.write('\n')

                # Check if there are right and left foot contacts
                if c_force_elt.getName()[-2:] == '_r':
                    nRightContacts += 1
                    rightFootContactBodies.append(geo1_frameName)
                    if not rightFootContact:
                        rightFootContact = True
                if c_force_elt.getName()[-2:] == '_l':
                    nLeftContacts += 1
                    leftFootContactBodies.append(geo1_frameName)
                    if not leftFootContact:
                        leftFootContact = True
        nContacts = nRightContacts + nLeftContacts
           
        # Compute residuals (joint torques).
        f.write('\t// Initialize system.\n')
        f.write('\tSimTK::State* state;\n')
        f.write('\tstate = new State(model->initSystem());\n\n')
    
        f.write('\t// Read inputs.\n')
        f.write('\tstd::vector<T> x(arg[0], arg[0] + NX);\n')
        f.write('\tstd::vector<T> u(arg[1], arg[1] + NU);\n')
        if treadmill:
            f.write('\tstd::vector<T> p(arg[2], arg[2] + 1);\n')
        f.write('\n')
        
        f.write('\t// States and controls.\n')
        if treadmill:
            f.write('\tT ua[NU_treadmill];\n')
            f.write('\tVector QsUs(NX_treadmill);\n')
        else:
            f.write('\tT ua[NU];\n')
            f.write('\tVector QsUs(NX);\n')        
        f.write('\t/// States\n')
        f.write('\tfor (int i = 0; i < NX; ++i) QsUs[i] = x[i];\n')
        if treadmill:
            f.write('\tQsUs[NX] = 0;\n')
            f.write('\tQsUs[NX+1] = p[0];\n')
        f.write('\t/// Controls\n')
        if treadmill:
            f.write('\tT ut[NU_treadmill];\n')
            f.write('\tfor (int i = 0; i < NU; ++i) ut[i] = u[i];\n')
            f.write('\tut[NU] = 0;\n')        
        f.write('\t/// OpenSim and Simbody have different state orders.\n')
        f.write('\tauto indicesOSInSimbody = getIndicesOpenSimInSimbody(*model);\n')
        if treadmill:
            f.write('\tfor (int i = 0; i < NU_treadmill; ++i) ua[i] = ut[indicesOSInSimbody[i]];\n\n')
        else:
            f.write('\tfor (int i = 0; i < NU; ++i) ua[i] = u[indicesOSInSimbody[i]];\n\n')
    
        f.write('\t// Set state variables and realize.\n')
        f.write('\tmodel->setStateVariableValues(*state, QsUs);\n')
        f.write('\tmodel->realizeVelocity(*state);\n\n')
        
        f.write('\t// Compute residual forces.\n')
        f.write('\t/// Set appliedMobilityForces (# mobilities).\n')
        if treadmill:
            f.write('\tVector appliedMobilityForces(nCoordinates_treadmill);\n')
        else:
            f.write('\tVector appliedMobilityForces(nCoordinates);\n')
        f.write('\tappliedMobilityForces.setToZero();\n')
        f.write('\t/// Set appliedBodyForces (# bodies + ground).\n')
        f.write('\tVector_<SpatialVec> appliedBodyForces;\n')
        f.write('\tint nbodies = model->getBodySet().getSize() + 1;\n')
        f.write('\tappliedBodyForces.resize(nbodies);\n')
        f.write('\tappliedBodyForces.setToZero();\n')
        f.write('\t/// Set gravity.\n')
        f.write('\tVec3 gravity(0);\n')
        f.write('\tgravity[1] = %.20f;\n' % model.get_gravity()[1])
        f.write('\t/// Add weights to appliedBodyForces.\n')
        f.write('\tfor (int i = 0; i < model->getBodySet().getSize(); ++i) {\n')
        f.write('\t\tmodel->getMatterSubsystem().addInStationForce(*state,\n')
        f.write('\t\tmodel->getBodySet().get(i).getMobilizedBodyIndex(),\n')
        f.write('\t\tmodel->getBodySet().get(i).getMassCenter(),\n')
        f.write('\t\tmodel->getBodySet().get(i).getMass()*gravity, appliedBodyForces);\n')
        f.write('\t}\n')    
        f.write('\t/// Add contact forces to appliedBodyForces.\n')
        
        count = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)     
            
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":
                c_force_elt_name = c_force_elt.getName()    
                
                f.write('\tArray<osim_double_adouble> Force_%s = %s->getRecordValues(*state);\n' % (str(count), c_force_elt_name))
                f.write('\tSpatialVec GRF_%s;\n' % (str(count)))           
                
                f.write('\tGRF_%s[0] = Vec3(Force_%s[3], Force_%s[4], Force_%s[5]);\n' % (str(count), str(count), str(count), str(count)))
                f.write('\tGRF_%s[1] = Vec3(Force_%s[0], Force_%s[1], Force_%s[2]);\n' % (str(count), str(count), str(count), str(count)))
                
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_frameName = geo1.getFrame().getName()
                
                f.write('\tint c_idx_%s = model->getBodySet().get("%s").getMobilizedBodyIndex();\n' % (str(count), geo1_frameName))            
                f.write('\tappliedBodyForces[c_idx_%s] += GRF_%s;\n' % (str(count), str(count)))
                count += 1
                f.write('\n')
                
        f.write('\t/// knownUdot.\n')
        if treadmill:
            f.write('\tVector knownUdot(nCoordinates_treadmill);\n')
        else:
            f.write('\tVector knownUdot(nCoordinates);\n')
        f.write('\tknownUdot.setToZero();\n')
        if treadmill:
            f.write('\tfor (int i = 0; i < nCoordinates_treadmill; ++i) knownUdot[i] = ua[i];\n')
        else:
            f.write('\tfor (int i = 0; i < nCoordinates; ++i) knownUdot[i] = ua[i];\n')
        f.write('\t/// Calculate residual forces.\n')
        if treadmill:
            f.write('\tVector residualMobilityForces(nCoordinates_treadmill);\n')
        else:
            f.write('\tVector residualMobilityForces(nCoordinates);\n')
        f.write('\tresidualMobilityForces.setToZero();\n')
        f.write('\tmodel->getMatterSubsystem().calcResidualForceIgnoringConstraints(*state,\n')
        f.write('\t\t\tappliedMobilityForces, appliedBodyForces, knownUdot, residualMobilityForces);\n\n')
            
        # Get body origins.
        f.write('\t/// Body origins.\n')
        for i in range(bodySet.getSize()):        
            c_body = bodySet.get(i)
            c_body_name = c_body.getName()            
            if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
                continue            
            f.write('\tVec3 %s_or = %s->getPositionInGround(*state);\n' % (c_body_name, c_body_name))
        f.write('\n')
            
        # Get GRFs.
        f.write('\t/// Ground reaction forces.\n')
        if rightFootContact:
            f.write('\tVec3 GRF_r(0);\n')
        if leftFootContact:
            f.write('\tVec3 GRF_l(0);\n')
        count = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)  
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":
                c_force_elt_name = c_force_elt.getName() 
                if c_force_elt_name[-2:] == "_r":
                    f.write('\tGRF_r += GRF_%s[1];\n'  % (str(count)))
                elif c_force_elt_name[-2:] == "_l":
                    f.write('\tGRF_l += GRF_%s[1];\n'  % (str(count)))
                else:
                    raise ValueError("Cannot identify contact side")
                count += 1
        f.write('\n')
            
        # Get GRMs.
        f.write('\t/// Ground reaction moments.\n')
        if rightFootContact:
            f.write('\tVec3 GRM_r(0);\n')
        if leftFootContact:
            f.write('\tVec3 GRM_l(0);\n')
        f.write('\tVec3 normal(0, 1, 0);\n\n')
        count = 0
        geo1_frameNames = []
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)  
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":
                c_force_elt_name = c_force_elt.getName() 
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_frameName = geo1.getFrame().getName() 
                
                if not geo1_frameName in geo1_frameNames:
                    f.write('\tSimTK::Transform TR_GB_%s = %s->getMobilizedBody().getBodyTransform(*state);\n' % (geo1_frameName, geo1_frameName))    
                    geo1_frameNames.append(geo1_frameName)
                    
                f.write('\tVec3 %s_location_G = %s->findStationLocationInGround(*state, %s_location);\n' % (c_force_elt_name, geo1_frameName, c_force_elt_name))                
                f.write('\tVec3 %s_locationCP_G = %s_location_G - %s_radius * normal;\n' % (c_force_elt_name, c_force_elt_name, c_force_elt_name))
                f.write('\tVec3 locationCP_G_adj_%i = %s_locationCP_G - 0.5*%s_locationCP_G[1] * normal;\n' % (count, c_force_elt_name, c_force_elt_name))
                f.write('\tVec3 %s_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_%i, *%s);\n' % (c_force_elt_name, count, geo1_frameName))
                f.write('\tVec3 GRM_%i = (TR_GB_%s*%s_locationCP_B) %% GRF_%s[1];\n' % (count, geo1_frameName, c_force_elt_name, str(count)))
                
                if c_force_elt_name[-2:] == "_r":
                    f.write('\tGRM_r += GRM_%i;\n'  % (count))   
                elif c_force_elt_name[-2:] == "_l": 
                    f.write('\tGRM_l += GRM_%i;\n'  % (count))   
                else:
                    raise ValueError("Cannot identify contact side")
                f.write('\n')                   
                count += 1
        
        # Save dict pointing to which elements are returned by F and in which
        # order, such as to facilitate using F when formulating problem.
        F_map = {}
        
        f.write('\t/// Outputs.\n')        
        # Export residuals (joint torques).
        f.write('\t/// Residual forces (OpenSim and Simbody have different state orders).\n')
        f.write('\tauto indicesSimbodyInOS = getIndicesSimbodyInOpenSim(*model);\n')
        f.write('\tfor (int i = 0; i < NU; ++i) res[0][i] =\n')
        f.write('\t\t\tvalue<T>(residualMobilityForces[indicesSimbodyInOS[i]]);\n')
        F_map['residuals'] = {}
        count = 0
        for coordinate in coordinates:
            if 'beta' in coordinate:
                continue
            F_map['residuals'][coordinate] = count 
            count += 1
        count_acc = nCoordinates
        
        # Export GRFs.
        f.write('\t/// Ground reaction forces.\n')        
        F_map['GRFs'] = {} 
        F_map['GRFs']['nContactSpheres'] = nContacts
        F_map['GRFs']['nRightContactSpheres'] = nRightContacts
        F_map['GRFs']['nLeftContactSpheres'] = nLeftContacts
        if rightFootContact:
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRF_r[i]);\n' % (count_acc))
            F_map['GRFs']['right'] = range(count_acc, count_acc+3)
            count_acc += 3
        if leftFootContact:
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRF_l[i]);\n' % (count_acc))
            F_map['GRFs']['left'] = range(count_acc, count_acc+3)
            count_acc += 3       
        
        # Export GRMs.
        f.write('\t/// Ground reaction moments.\n')
        F_map['GRMs'] = {}
        if rightFootContact:
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRM_r[i]);\n' % (count_acc))
            F_map['GRMs']['right'] = range(count_acc, count_acc+3)
            count_acc += 3
        if leftFootContact:
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRM_l[i]);\n' % (count_acc))
            F_map['GRMs']['left'] = range(count_acc, count_acc+3)
            count_acc += 3
        
        # Export individual GRFs.
        f.write('\t/// Ground reaction forces per sphere.\n')
        count = 0
        F_map['GRFs']['rightContactSpheres'] = []
        F_map['GRFs']['leftContactSpheres'] = []
        F_map['GRFs']['rightContactSphereBodies'] = rightFootContactBodies
        F_map['GRFs']['leftContactSphereBodies'] = leftFootContactBodies        
        for i in range(forceSet.getSize()):
            c_force_elt = forceSet.get(i) 
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":
                f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRF_%i[1][i]);\n' % (count_acc, count))
                F_map['GRFs'][c_force_elt.getName()] = range(count_acc, count_acc+3)
                if c_force_elt.getName()[-2:] == "_r":
                    F_map['GRFs']['rightContactSpheres'].append(c_force_elt.getName())
                elif c_force_elt.getName()[-2:] == "_l":
                    F_map['GRFs']['leftContactSpheres'].append(c_force_elt.getName())
                count += 1
                count_acc += 3
        f.write('\n')
        
        # Export individual contact locations.
        f.write('\t/// Contact point locations per sphere.\n')
        F_map['COPs'] = {}
        count = 0
        for i in range(forceSet.getSize()):
            c_force_elt = forceSet.get(i) 
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":
                f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(locationCP_G_adj_%i[i]);\n' % (count_acc, count))
                F_map['COPs'][c_force_elt.getName()] = range(count_acc, count_acc+3)
                count += 1
                count_acc += 3
        f.write('\n')
        
        # Export body origins.
        f.write('\t/// Body origins.\n')
        F_map['body_origins'] = {}
        count = 0
        for i in range(bodySet.getSize()):        
            c_body = bodySet.get(i)
            c_body_name = c_body.getName()
            if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
                continue
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(%s_or[i]);\n' % (count_acc+count*3, c_body_name))
            F_map['body_origins'][c_body_name] = range(count_acc+count*3, count_acc+count*3+3)
            count += 1
        count_acc += 3*count
            
        f.write('\n')
        f.write('\treturn 0;\n')
        f.write('}\n\n')
        
        # Residuals (joint torques), 3D GRFs (combined), 3D GRMs (combined),
        # 3D GRFs (per sphere), 3D COP (per sphere), and 3D body origins.
        nOutputs = nCoordinates + 3*(2*nContacts+nBodies)
        if rightFootContact:
            nOutputs += 2*3
        if leftFootContact:
            nOutputs += 2*3
        f.write('constexpr int NR = %i; \n\n' % (nOutputs))
        
        f.write('int main() {\n')
        f.write('\tRecorder x[NX];\n')
        f.write('\tRecorder u[NU];\n')
        if treadmill:
            f.write('\tRecorder p[1];\n')            
        f.write('\tRecorder tau[NR];\n')
        f.write('\tfor (int i = 0; i < NX; ++i) x[i] <<= 0;\n')
        f.write('\tfor (int i = 0; i < NU; ++i) u[i] <<= 0;\n')
        if treadmill:
            f.write('\tp[0] <<= 0;\n')
            f.write('\tconst Recorder* Recorder_arg[n_in] = { x,u,p };\n')
        else:
            f.write('\tconst Recorder* Recorder_arg[n_in] = { x,u };\n')
        f.write('\tRecorder* Recorder_res[n_out] = { tau };\n')
        f.write('\tF_generic<Recorder>(Recorder_arg, Recorder_res);\n')
        f.write('\tdouble res[NR];\n')
        f.write('\tfor (int i = 0; i < NR; ++i) Recorder_res[0][i] >>= res[i];\n')
        f.write('\tRecorder::stop_recording();\n')
        f.write('\treturn 0;\n')
        f.write('}\n')
        
        # Save dict.
        np.save(pathOutputMap, F_map)
            
    # %% Build external Function.
    if build_externalFunction:
        pathDCAD = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD') 
        buildExternalFunction(
            externalFunctionName, pathDCAD, pathOutputExternalFunctionFolder,
            3*nCoordinates, treadmill=treadmill, 
            useExpressionGraphFunction=useExpressionGraphFunction)
        
    # %% Verification.
    if verifyID:
        # Run ID with the .osim file
        pathGenericTemplates = os.path.join(baseDir, "OpenSimPipeline") 
        pathGenericIDFolder = os.path.join(pathGenericTemplates,
                                           "InverseDynamics")
        pathGenericIDSetupFile = os.path.join(pathGenericIDFolder, 
                                              "Setup_InverseDynamics.xml")
        idTool = opensim.InverseDynamicsTool(pathGenericIDSetupFile)
        idTool.setName("ID_withOsimAndIDTool")
        idTool.setModelFileName(pathModel)
        idTool.setResultsDir(pathOutputExternalFunctionFolder)
        idTool.setCoordinatesFileName(os.path.join(
            pathGenericIDFolder, "DefaultPosition_rajagopal.mot"))
        idTool.setOutputGenForceFileName("ID_withOsimAndIDTool.sto")       
        pathSetupID = os.path.join(pathOutputExternalFunctionFolder, 
                                   "Setup_InverseDynamics.xml")
        idTool.printToXML(pathSetupID)
        idTool.run()
        
        # Extract torques from .osim + ID tool.    
        headers = []
        nCoordinatesAll = coordinateSet.getSize()
        for coord in range(nCoordinatesAll):                
            if (coordinateSet.get(coord).getName() == "pelvis_tx" or 
                coordinateSet.get(coord).getName() == "pelvis_ty" or 
                coordinateSet.get(coord).getName() == "pelvis_tz" or
                coordinateSet.get(coord).getName() == "knee_angle_r_beta" or 
                coordinateSet.get(coord).getName() == "knee_angle_l_beta"):
                suffix_header = "_force"
            else:
                suffix_header = "_moment"
            headers.append(coordinateSet.get(coord).getName() + suffix_header)
            
        ID_osim_df = storage_to_dataframe(os.path.join(
            pathOutputExternalFunctionFolder,"ID_withOsimAndIDTool.sto"), 
            headers)
        ID_osim = np.zeros((nCoordinates))
        count = 0
        for coordinate in coordinates:
            if (coordinate == "pelvis_tx" or 
                coordinate == "pelvis_ty" or 
                coordinate == "pelvis_tz"):
                suffix_header = "_force"
            else:
                suffix_header = "_moment"
            if 'beta' in coordinate:
                continue                
            ID_osim[count] = ID_osim_df.iloc[0][coordinate + suffix_header]
            count += 1
        
        # Extract torques from external function.        
        vec1 = np.zeros((nCoordinates*2, 1))
        vec1[::2, :] = 0.05   
        vec1[8, :] = -0.05
        vec2 = np.zeros((nCoordinates, 1))        
        if treadmill:
            vec4 = np.zeros((1, 1))
            vec3 = np.concatenate((vec1,vec2,vec4))
        else:            
            vec3 = np.concatenate((vec1,vec2))

        if useExpressionGraphFunction:
            # Approach 1: Expression graph.
            pathExpressionGraph = os.path.join(
                pathOutputExternalFunctionFolder, externalFunctionName + '.py')
            if os.path.exists(pathExpressionGraph):
                dim = vec3.shape[0]
                sys.path.append(pathOutputExternalFunctionFolder)
                os.chdir(pathOutputExternalFunctionFolder)
                F = getF_expressingGraph(dim, externalFunctionName)
                sys.path.remove(pathOutputExternalFunctionFolder)
                os.chdir(pathCWD)
                ID_F = (F(vec3)).full().flatten()[:nCoordinates]
                assert(np.max(np.abs(ID_osim - ID_F)) < 1e-6), (
                    'error F vs ID tool {}'.format(np.max(np.abs(ID_osim - ID_F))))
                print('Verification torque with expression graph: success')

        else:
            # Approach 2: External function.
            # Will be deprecated in the future.
            os_system = platform.system()
            if os_system == 'Windows':
                F_ext = '.dll'
            elif os_system == 'Linux':
                F_ext = '.so'
            elif os_system == 'Darwin':
                F_ext = '.dylib'
            pathExternalFunction = os.path.join(
                pathOutputExternalFunctionFolder, externalFunctionName + F_ext)
            if os.path.exists(pathExternalFunction):
                import casadi as ca            
                F = ca.external('F', pathExternalFunction)
                ID_F = (F(vec3)).full().flatten()[:nCoordinates]
                assert(np.max(np.abs(ID_osim - ID_F)) < 1e-6), (
                    'error F vs ID tool {}'.format(np.max(np.abs(ID_osim - ID_F))))
                print('Verification torque with external function: success')
        
        # Clean up.
        os.remove(os.path.join(
            pathOutputExternalFunctionFolder,"ID_withOsimAndIDTool.sto"))
        os.remove(os.path.join(
            pathOutputExternalFunctionFolder,"Setup_InverseDynamics.xml"))

# %% Generate c code from expression graph.
def generateF(dim, script_name):
    foo_module = importlib.import_module(script_name)
    importlib.reload(foo_module)
    cg = ca.CodeGenerator('foo_jac')
    arg = ca.SX.sym('arg', dim)
    y,_,_ = foo_module.foo(arg)
    F = ca.Function('F',[arg],[y])
    cg.add(F)
    cg.add(F.jacobian())
    cg.generate()
    
# %% Generate function from expression graph.
def getF_expressingGraph(dim, script_name):
    foo_module = importlib.import_module(script_name)
    importlib.reload(foo_module)
    arg = ca.SX.sym('arg', dim)
    y,_,_ = foo_module.foo(arg)
    F = ca.Function('F',[arg],[y])    
    return F    

# %% Compile external function.
def buildExternalFunction(filename, pathDCAD, CPP_DIR, nInputs,
                          treadmill=False, useExpressionGraphFunction=True):       
    
    # %% Part 1: build expression graph (i.e., generate foo.py).
    pathMain = os.getcwd()
    pathBuildExpressionGraph = os.path.join(pathDCAD, 'buildExpressionGraph')
    pathBuild = os.path.join(pathDCAD, 'build-ExpressionGraph' + filename)
    os.makedirs(pathBuild, exist_ok=True)
    OpenSimAD_DIR = os.path.join(pathDCAD, 'opensimAD-install')
    os.makedirs(OpenSimAD_DIR, exist_ok=True)
    os_system = platform.system()
    
    if os_system == 'Windows':
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'windows')        
        BIN_DIR = os.path.join(OpenSimADOS_DIR, 'bin')
        SDK_DIR = os.path.join(OpenSimADOS_DIR, 'sdk')
        # Download libraries if not existing locally.
        if not os.path.exists(BIN_DIR):
            url = 'https://sourceforge.net/projects/opensimad/files/windows.zip'
            zipfilename = 'windows.zip'
            try:
                download_file(url, zipfilename)
            except:
                try:
                    download_file_2(url, zipfilename)
                except:
                    error_msg = """ \n\n\n
                    Problem when downloading third-party libraries. You can download them manually:
                        1. Download the zip file hosted here: {},
                        2. Extract the files, and
                        3. Copy then under: <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install.
                    You should have:
                        1. <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install/windows/bin and
                        2. <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install/windows/sdk \n\n\n""".format(url)
                    raise ValueError(error_msg)                    
            with zipfile.ZipFile('windows.zip', 'r') as zip_ref:
                zip_ref.extractall(OpenSimAD_DIR)
            os.remove('windows.zip')

        # Get the list of available generators
        available_generators_output = subprocess.check_output(['cmake', '--help'], universal_newlines=True)
        available_generators = available_generators_output.splitlines()

        # Select the preferred generator if it is available
        preferred_generator = get_preferred_generator(available_generators)
        if not preferred_generator:
            raise ValueError("Generator not found. Have you installed Visual Studio with the C++ extension? See instructions: https://github.com/stanfordnmbl/opencap-processing?tab=readme-ov-file#muscle-driven-simulations")

        cmd1 = 'cmake "' + pathBuildExpressionGraph + '" -G "' + preferred_generator + '" -A x64 -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + SDK_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "cmake --build . --config RelWithDebInfo"
        
    elif os_system == 'Linux':
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'linux')
        # Download libraries if not existing locally.
        if not os.path.exists(os.path.join(OpenSimAD_DIR, 'linux', 'lib')):
            url = 'https://sourceforge.net/projects/opensimad/files/linux.tar.gz'
            zipfilename = 'linux.tar.gz'                
            try:
                download_file(url, zipfilename)
            except:
                try:
                    download_file_2(url, zipfilename)
                except:
                    error_msg = """ \n\n\n
                    Problem when downloading third-party libraries. You can download them manually:
                        1. Download the tar file hosted here: {},
                        2. Extract the files, and
                        3. Copy then under: <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install.
                    You should have:
                        1. <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install/linux/lib and
                        2. <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install/linux/include \n\n\n""".format(url)
                    raise ValueError(error_msg) 
            cmd_tar = 'tar -xf linux.tar.gz -C "{}"'.format(OpenSimAD_DIR)
            os.system(cmd_tar)
            os.remove('linux.tar.gz')
        cmd1 = 'cmake "' + pathBuildExpressionGraph + '" -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + OpenSimADOS_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "make"
        BIN_DIR = pathBuild
        
    elif os_system == 'Darwin':
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'macOS')
        # Download libraries if not existing locally.
        if not os.path.exists(os.path.join(OpenSimAD_DIR, 'macOS', 'lib')):
            url = 'https://sourceforge.net/projects/opensimad/files/macOS.tgz'
            zipfilename = 'macOS.tgz'                
            try:
                download_file(url, zipfilename)
            except:
                try:
                    download_file_2(url, zipfilename)
                except:
                    error_msg = """ \n\n\n
                    Problem when downloading third-party libraries. You can download them manually:
                        1. Download the tar file hosted here: {},
                        2. Extract the files, and
                        3. Copy then under: <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install.
                    You should have:
                        1. <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install/macOS/lib and
                        2. <local_path>/opencap-processing/UtilsDynamicSimulations/OpenSimAD/opensimAD-install/macOS/include \n\n\n""".format(url)
                    raise ValueError(error_msg) 
            cmd_tar = 'tar -xf macOS.tgz -C "{}"'.format(OpenSimAD_DIR)
            os.system(cmd_tar)
            os.remove('macOS.tgz')
        cmd1 = 'cmake -DCMAKE_OSX_ARCHITECTURES="x86_64" "' + pathBuildExpressionGraph + '" -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + OpenSimADOS_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "make"
        BIN_DIR = pathBuild
    
    os.chdir(pathBuild)    
    os.system(cmd1)    
    os.system(cmd2)
    
    if os_system == 'Windows':
        os.chdir(BIN_DIR)
        path_EXE = os.path.join(pathBuild, 'RelWithDebInfo', filename + '.exe')
        cmd2w = '"{}"'.format(path_EXE)
        os.system(cmd2w)

    fooName = "foo"
    path_external_filename_foo = os.path.join(BIN_DIR, fooName + '.py')
    if useExpressionGraphFunction:
        shutil.copy2(path_external_filename_foo, CPP_DIR)
        os.rename(os.path.join(CPP_DIR, fooName + '.py'), os.path.join(CPP_DIR, filename + '.py'))

    # %% Part 2: build external function (i.e., build .dll/.so/.dylib).
    else:    
        pathBuildExternalFunction = os.path.join(pathDCAD, 'buildExternalFunction')    
        path_external_functions_filename_build = os.path.join(pathDCAD, 'build-ExternalFunction' + filename)
        path_external_functions_filename_install = os.path.join(pathDCAD, 'install-ExternalFunction' + filename)
        os.makedirs(path_external_functions_filename_build, exist_ok=True) 
        os.makedirs(path_external_functions_filename_install, exist_ok=True)
        shutil.copy2(path_external_filename_foo, pathBuildExternalFunction)    
        
        sys.path.append(pathBuildExternalFunction)
        os.chdir(pathBuildExternalFunction)
        
        if treadmill:
            generateF(nInputs+1, fooName)
        else:
            generateF(nInputs, fooName)
        
        if os_system == 'Windows':
            cmd3 = 'cmake "' + pathBuildExternalFunction + '" -G "' + preferred_generator + '" -A x64 -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
            cmd4 = "cmake --build . --config RelWithDebInfo --target install"
        elif os_system == 'Linux':
            cmd3 = 'cmake "' + pathBuildExternalFunction + '" -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
            cmd4 = "make install"
        elif os_system == 'Darwin':
            cmd3 = 'cmake "' + pathBuildExternalFunction + '" -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
            cmd4 = "make install"
        
        os.chdir(path_external_functions_filename_build)
        os.system(cmd3)
        os.system(cmd4)    
        os.chdir(pathMain)
        
        if os_system == 'Windows':
            shutil.copy2(os.path.join(path_external_functions_filename_install, 'bin', filename + '.dll'), CPP_DIR)
        elif os_system == 'Linux':
            shutil.copy2(os.path.join(path_external_functions_filename_install, 'lib', 'lib' + filename + '.so'), CPP_DIR)
            os.rename(os.path.join(CPP_DIR, 'lib' + filename + '.so'), os.path.join(CPP_DIR, filename + '.so'))
        elif os_system == 'Darwin':
            shutil.copy2(os.path.join(path_external_functions_filename_install, 'lib', 'lib' + filename + '.dylib'), CPP_DIR)
            os.rename(os.path.join(CPP_DIR, 'lib' + filename + '.dylib'), os.path.join(CPP_DIR, filename + '.dylib'))
        
        os.remove(os.path.join(pathBuildExternalFunction, "foo_jac.c"))        
        os.remove(os.path.join(pathBuildExternalFunction, fooName + '.py'))
        shutil.rmtree(pathBuild)
        shutil.rmtree(path_external_functions_filename_install)
        shutil.rmtree(path_external_functions_filename_build)
    if os_system == 'Windows':
        os.remove(path_external_filename_foo)

def get_preferred_generator(available_generators):
    # Define a list of preferred generators in order of preference
    preferred_generators = [
        "Visual Studio 17 2022",
        "Visual Studio 16 2019",
        "Visual Studio 15 2017"
    ]
    
    # Loop through each preferred generator and look for matches
    matches = {}
    for generator in preferred_generators:        
        for line in available_generators:
            if re.search(r'\b' + re.escape(generator) + r'\b', line):
                matches[generator] = line
        
    # If there are matches, prioritize the one with an asterisk
    if matches:
        for generator in matches:
            if '*' in matches[generator]:
                return generator
                
        # If no match with asterisk is found, return the first match        
        return list(matches.keys())[0]
    
    return None
    
# %% Download file given url (approach 1).
def download_file(url, file_name):
    
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
# %% Download file given url (approach 2).
def download_file_2(url, file_name):
    
    response = requests.get(url)
    open(file_name, 'wb').write(response.content)
    
# %% Plot results simulations.
# TODO: simplify and clean up.
def plotResultsOpenSimAD(dataDir, subject, motion_filename, settings,
                         cases=['default'], mainPlots=True):
    
    # %% Load optimal trajectories.
    pathOSData = os.path.join(dataDir, subject, 'OpenSimData')
    suff_path = ''
    if 'repetition' in settings:
        suff_path = '_rep' + str(settings['repetition'])
    c_pathResults = os.path.join(pathOSData, 'Dynamics', 
                                 motion_filename + suff_path)    
    c_tr = np.load(os.path.join(c_pathResults, 'optimaltrajectories.npy'),
                   allow_pickle=True).item()    
    optimaltrajectories = {}
    for case in cases:
        optimaltrajectories[case] = c_tr[case]
        
    colors = sns.color_palette('colorblind', len(cases))

    # %% Plot settings
    linewidth = 2
    fontsizeLabel = 14
    fontsizeTitle = 14
    fontsizeTicks = 14
    fontsizeLegend = 14
    fontsizeSubTitle = 16
    
    # %% Joint coordinates.
    joints = optimaltrajectories[cases[0]]['coordinates']
    nJoints = len(joints)
    rotationalJoints = optimaltrajectories[cases[0]]['rotationalCoordinates']
    ny = np.ceil(np.sqrt(nJoints))
    fig, axs = plt.subplots(int(ny), int(ny))
    fig.suptitle('Coordinate values', fontsize=fontsizeSubTitle, fontweight='bold')
    for i, ax in enumerate(axs.flat):
        if i < nJoints:
            if joints[i] in rotationalJoints:
                scale_angles = 180 / np.pi
            else:
                scale_angles = 1
            plotReference = False
            plotMocapReference = False
            for c, case in enumerate(cases):
                if joints[i] in optimaltrajectories[case]['coordinates']:                        
                    idx_coord = optimaltrajectories[case]['coordinates'].index(joints[i])
                    if 'coordinate_values_mocap' in optimaltrajectories[case]:
                        if not plotMocapReference:
                            ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                    optimaltrajectories[case]['coordinate_values_mocap'][idx_coord:idx_coord+1,:].T * scale_angles, c='black', linestyle='dashed', label='Mocap-based IK', linewidth=linewidth)
                            plotMocapReference = True                
                    if not plotReference:
                        ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                optimaltrajectories[case]['coordinate_values_toTrack'][idx_coord:idx_coord+1,:].T * scale_angles, c='black', linestyle='dotted', label='Tracked data (OpenCap): ' + cases[c], linewidth=linewidth)
                        plotReference = True
                    ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                            optimaltrajectories[case]['coordinate_values'][idx_coord:idx_coord+1,:-1].T * scale_angles, c=colors[c], label='Dynamic simulation: ' + cases[c], linewidth=linewidth)   
            ax.set_title(joints[i], fontsize=fontsizeTitle, fontweight='bold')
            handles, labels = ax.get_legend_handles_labels()
    fig.align_ylabels()
    # Remove empty subplots.
    for i in range(nJoints, int(ny)**2):
        fig.delaxes(axs.flatten()[i])
    # Remove top and right spines.
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fontsizeTicks)
    # Add labels
    mm = int(ny)*(int(ny)-1) -(int(ny)*int(ny)-nJoints)
    for i in range(mm,nJoints):
        axs.flatten()[i].set_xlabel('Time (s)', fontsize=fontsizeLabel, fontweight='bold')
    for ax in axs[0:, 0]:
        ax.set_ylabel('(deg)', fontsize=fontsizeLabel, fontweight='bold')
    for ax in axs[0, 3:4]:
        ax.set_ylabel('(m)', fontsize=fontsizeLabel, fontweight='bold')    
    # Add legend.
    fig.legend(handles, labels, loc='upper right', fontsize=fontsizeLegend)
    # Change subplot spacing.
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # Clean up ticks and labels.    
    for i in range(0, mm):
        axs.flatten()[i].set_xticklabels([])
    plt.show()
        
    # %% Joint speeds.
    if not mainPlots:
        fig, axs = plt.subplots(int(ny), int(ny))
        fig.suptitle('Coordinate speeds', fontsize=fontsizeSubTitle, fontweight='bold')
        for i, ax in enumerate(axs.flat):
            if i < nJoints:
                if joints[i] in rotationalJoints:
                    scale_angles = 180 / np.pi
                else:
                    scale_angles = 1
                plotReference = False
                plotMocapReference = False
                for c, case in enumerate(cases):
                    if joints[i] in optimaltrajectories[case]['coordinates']:                        
                        idx_coord = optimaltrajectories[case]['coordinates'].index(joints[i])
                        if 'coordinate_speeds_mocap' in optimaltrajectories[case]:
                            if not plotMocapReference:
                                ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                        optimaltrajectories[case]['coordinate_speeds_mocap'][idx_coord:idx_coord+1,:].T * scale_angles, c='black', linestyle='dashed', label='Mocap-based IK', linewidth=linewidth)
                                plotMocapReference = True
                        if not plotReference:
                            ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                    optimaltrajectories[case]['coordinate_speeds_toTrack'][idx_coord:idx_coord+1,:].T * scale_angles, c='black', linestyle='dotted', label='Tracked data (OpenCap): ' + cases[c], linewidth=linewidth)
                            plotReference = True
                        ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                optimaltrajectories[case]['coordinate_speeds'][idx_coord:idx_coord+1,:-1].T * scale_angles, c=colors[c], label='Dynamic simulation: ' + cases[c], linewidth=linewidth)   
                ax.set_title(joints[i], fontsize=fontsizeTitle, fontweight='bold')
                handles, labels = ax.get_legend_handles_labels()
        fig.align_ylabels()
        # Remove empty subplots.
        for i in range(nJoints, int(ny)**2):
            fig.delaxes(axs.flatten()[i])
        # Remove top and right spines.
        for ax in axs.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=fontsizeTicks)
        # Add labels
        mm = int(ny)*(int(ny)-1) -(int(ny)*int(ny)-nJoints)
        for i in range(mm,nJoints):
            axs.flatten()[i].set_xlabel('Time (s)', fontsize=fontsizeLabel, fontweight='bold')
        for ax in axs[0:, 0]:
            ax.set_ylabel('(deg/s)', fontsize=fontsizeLabel, fontweight='bold')
        for ax in axs[0, 3:4]:
            ax.set_ylabel('(m/s)', fontsize=fontsizeLabel, fontweight='bold')  
        # Add legend.
        fig.legend(handles, labels, loc='upper right', fontsize=fontsizeLegend)
        # Change subplot spacing.
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # Clean up ticks and labels.
        for i in range(0, mm):
            axs.flatten()[i].set_xticklabels([])
        plt.show()
        
    # %% Joint accelerations.
    if not mainPlots:
        fig, axs = plt.subplots(int(ny), int(ny))
        fig.suptitle('Coordinate accelerations', fontsize=fontsizeSubTitle, fontweight='bold')
        for i, ax in enumerate(axs.flat):
            if i < nJoints:
                if joints[i] in rotationalJoints:
                    scale_angles = 180 / np.pi
                else:
                    scale_angles = 1
                plotReference = False
                plotMocapReference = False
                for c, case in enumerate(cases):
                    if joints[i] in optimaltrajectories[case]['coordinates']:                        
                        idx_coord = optimaltrajectories[case]['coordinates'].index(joints[i])
                        if 'coordinate_accelerations_mocap' in optimaltrajectories[case]:
                            if not plotMocapReference:
                                ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                        optimaltrajectories[case]['coordinate_accelerations_mocap'][idx_coord:idx_coord+1,:].T * scale_angles, c='black', linestyle='dashed', label='Mocap-based IK', linewidth=linewidth)
                                plotMocapReference = True
                        if not plotReference:
                            ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                    optimaltrajectories[case]['coordinate_accelerations_toTrack'][idx_coord:idx_coord+1,:].T * scale_angles, c='black', linestyle='dotted', label='Tracked data (OpenCap): ' + cases[c], linewidth=linewidth)
                            plotReference = True
                        ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                optimaltrajectories[case]['coordinate_accelerations'][idx_coord:idx_coord+1,:].T * scale_angles, c=colors[c], label='Dynamic simulation: ' + cases[c], linewidth=linewidth)   
                ax.set_title(joints[i], fontsize=fontsizeTitle, fontweight='bold')
                handles, labels = ax.get_legend_handles_labels()
        fig.align_ylabels()
        # Remove empty subplots.
        for i in range(nJoints, int(ny)**2):
            fig.delaxes(axs.flatten()[i])
        # Remove top and right spines.
        for ax in axs.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=fontsizeTicks)
        # Add labels
        mm = int(ny)*(int(ny)-1) -(int(ny)*int(ny)-nJoints)
        for i in range(mm,nJoints):
            axs.flatten()[i].set_xlabel('Time (s)', fontsize=fontsizeLabel, fontweight='bold')
        for ax in axs[0:, 0]:
            ax.set_ylabel('(deg/s2)', fontsize=fontsizeLabel, fontweight='bold')
        for ax in axs[0, 3:4]:
            ax.set_ylabel('(m/s2)', fontsize=fontsizeLabel, fontweight='bold')  
        # Add legend.
        fig.legend(handles, labels, loc='upper right', fontsize=fontsizeLegend)
        # Change subplot spacing.
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # Clean up ticks and labels.
        for i in range(0, mm):
            axs.flatten()[i].set_xticklabels([])
        plt.show()
        
    # %% Joint torques.
    fig, axs = plt.subplots(int(ny), int(ny))
    fig.suptitle('Joint torques', fontsize=fontsizeSubTitle, fontweight='bold')
    for i, ax in enumerate(axs.flat):
        if i < nJoints:
            plotReference = False
            for c, case in enumerate(cases):
                if joints[i] in optimaltrajectories[case]['coordinates']:                        
                    idx_coord = optimaltrajectories[case]['coordinates'].index(joints[i])
                    if 'torques_mocap' in optimaltrajectories[case]:
                        if not plotReference:
                            ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                    optimaltrajectories[case]['torques_mocap'][idx_coord:idx_coord+1,:].T, c='black', linestyle='dashed', label='Mocap-based ID', linewidth=linewidth)
                            plotReference = True
                    ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                            optimaltrajectories[case]['torques'][idx_coord:idx_coord+1,:].T, c=colors[c], label='Dynamic simulation: ' + cases[c], linewidth=linewidth)     
            ax.set_title(joints[i], fontsize=fontsizeTitle, fontweight='bold')
            handles, labels = ax.get_legend_handles_labels()
    fig.align_ylabels()
    
    # Remove empty subplots.
    for i in range(nJoints, int(ny)**2):
        fig.delaxes(axs.flatten()[i])
    # Remove top and right spines.
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fontsizeTicks)
    # Add labels
    mm = int(ny)*(int(ny)-1) -(int(ny)*int(ny)-nJoints)
    for i in range(mm,nJoints):
        axs.flatten()[i].set_xlabel('Time (s)', fontsize=fontsizeLabel, fontweight='bold')
    for ax in axs[:, 0]:
        ax.set_ylabel('(Nm)', fontsize=fontsizeLabel, fontweight='bold')
    for ax in axs[0, 3:4]:
        ax.set_ylabel('(N)', fontsize=fontsizeLabel, fontweight='bold') 
    # Add legend.
    fig.legend(handles, labels, loc='upper right', fontsize=fontsizeLegend)
    # Change subplot spacing.
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # Clean up ticks and labels.
    for i in range(0, mm):
        axs.flatten()[i].set_xticklabels([])
    plt.show()
        
    # %% GRFs.
    GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
    NGRF = len(GRF_labels)
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Ground reaction forces', fontsize=fontsizeSubTitle, fontweight='bold')
    for i, ax in enumerate(axs.flat):
        if i < NGRF:
            plotedGRF = False
            for c, case in enumerate(cases):
                if 'GRF_experimental' in optimaltrajectories[case] and not plotedGRF:
                    plotedGRF = True
                    ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                            optimaltrajectories[case]['GRF_experimental'][i:i+1,:].T, c='black', linestyle='dashed', label='Force plate', linewidth=linewidth)   
                ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                        optimaltrajectories[case]['GRF'][i:i+1,:].T, c=colors[c], label='Dynamic simulation: ' + cases[c], linewidth=linewidth)          
            ax.set_title(GRF_labels[i], fontsize=fontsizeTitle, fontweight='bold')
            handles, labels = ax.get_legend_handles_labels()
    fig.align_ylabels()    
    # Remove top and right spines.
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fontsizeTicks)
    # Add labels
    for ax in axs[-1, :]:
        ax.set_xlabel('Time (s)', fontsize=fontsizeLabel, fontweight='bold')
    for ax in axs[:, 0]:
        ax.set_ylabel('(N)', fontsize=fontsizeLabel, fontweight='bold')
    # Add legend.
    fig.legend(handles, labels, loc='upper right', fontsize=fontsizeLegend)
    # Change subplot spacing.
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
        
    # %% GRMs.
    if not mainPlots:
        GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
        NGRF = len(GRF_labels)
        fig, axs = plt.subplots(2, 3)
        fig.suptitle('Ground reaction moments', fontsize=fontsizeSubTitle, fontweight='bold')
        for i, ax in enumerate(axs.flat):
            if i < NGRF:
                plotedGRF = False
                for c, case in enumerate(cases):
                    if 'GRM_experimental' in optimaltrajectories[case] and not plotedGRF:
                        plotedGRF = True
                        ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                optimaltrajectories[case]['GRM_experimental'][i:i+1,:].T, c='black', linestyle='dashed', label='Force plate', linewidth=linewidth)   
                    ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                            optimaltrajectories[case]['GRM'][i:i+1,:].T, c=colors[c], label='Dynamic simulation: ' + cases[c], linewidth=linewidth)          
                ax.set_title(GRF_labels[i], fontsize=fontsizeTitle, fontweight='bold')
                handles, labels = ax.get_legend_handles_labels()
        fig.align_ylabels()    
        # Remove top and right spines.
        for ax in axs.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=fontsizeTicks)
        # Add labels
        for ax in axs[-1, :]:
            ax.set_xlabel('Time (s)', fontsize=fontsizeLabel, fontweight='bold')
        for ax in axs[:, 0]:
            ax.set_ylabel('(Nm)', fontsize=fontsizeLabel, fontweight='bold')
        # Add legend.
        fig.legend(handles, labels, loc='upper right', fontsize=fontsizeLegend)
        # Change subplot spacing.
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()
        
    # %% Muscle activations.
    plotMuscleActivations = False
    for case in cases:
        if 'muscle_activations' in optimaltrajectories[case]:
            plotMuscleActivations = True

    if not plotMuscleActivations:
        return

    muscles = optimaltrajectories[cases[0]]['muscles']
    NMuscles = len(muscles)
    ny = np.ceil(np.sqrt(NMuscles))
    fig, axs = plt.subplots(int(ny), int(ny))
    fig.suptitle('Muscle activations', fontsize=fontsizeSubTitle, fontweight='bold') 
    for i, ax in enumerate(axs.flat):
        if i < NMuscles:
            plotedEMG = False
            for c, case in enumerate(cases):
                if 'muscle_activations_emg' in optimaltrajectories[case] and not plotedEMG:
                    c_emg = optimaltrajectories[case]['muscle_activations_emg'][i:i+1,:].T
                    # check if there is any value different from nan
                    if not np.isnan(c_emg).all():                        
                        plotedEMG = True
                        ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                                optimaltrajectories[case]['muscle_activations_emg'][i:i+1,:].T, c='black', linestyle='dashed', label='EMG', linewidth=linewidth)
                if 'muscle_activations' in optimaltrajectories[case]:
                    ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
                            optimaltrajectories[case]['muscle_activations'][i:i+1,:-1].T, c=colors[c], label='Dynamic simulation: ' + cases[c], linewidth=linewidth)         
            ax.set_title(muscles[i], fontsize=fontsizeTitle, fontweight='bold')
            ax.set_ylim((0,1))
            handles, labels = ax.get_legend_handles_labels()
    fig.align_ylabels()
    
    # Remove empty subplots.
    for i in range(NMuscles, int(ny)**2):
        fig.delaxes(axs.flatten()[i])
    # Remove top and right spines.
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fontsizeTicks)
    # Add labels
    mm = int(ny)*(int(ny)-1) -(int(ny)*int(ny)-NMuscles)
    for i in range(mm,NMuscles):
        axs.flatten()[i].set_xlabel('Time (s)', fontsize=fontsizeLabel, fontweight='bold')
    for ax in axs[:, 0]:
        ax.set_ylabel('()', fontsize=fontsizeLabel, fontweight='bold')
    # Add legend.
    fig.legend(handles, labels, loc='upper right', fontsize=fontsizeLegend)
    # Change subplot spacing.
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # Clean up ticks and labels.
    for i in range(0, mm):
        axs.flatten()[i].set_xticklabels([])
    plt.show()
    
    # # %% Joint torques: breakdown.  
    # if not mainPlots:
    #     muscleDrivenJoints = optimaltrajectories[cases[0]][
    #         'muscle_driven_joints']
    #     nMuscleDrivenJoints = len(muscleDrivenJoints)
    #     ny = np.ceil(np.sqrt(nMuscleDrivenJoints))
    #     fig, axs = plt.subplots(3, 4, sharex=True)
    #     fig.suptitle('Net torque contributions') 
    #     for i, ax in enumerate(axs.flat):
    #         if i < nMuscleDrivenJoints:
    #             color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
    #             for c, case in enumerate(cases):
    #                 c_col = next(color)
    #                 if muscleDrivenJoints[i] in optimaltrajectories[case]['muscle_driven_joints']:                        
    #                     idx_coord = optimaltrajectories[case]['muscle_driven_joints'].index(muscleDrivenJoints[i])
    #                     ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
    #                             optimaltrajectories[case]['passive_muscle_torques'][idx_coord:idx_coord+1,:].T, c=c_col, linestyle='dashed', label='passive muscle torque ' + cases[c])
    #                     ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
    #                             optimaltrajectories[case]['active_muscle_torques'][idx_coord:idx_coord+1,:].T, c=c_col, linestyle='solid', label='active muscle torque ' + cases[c])                
    #                     c_sum = optimaltrajectories[case]['passive_muscle_torques'][idx_coord:idx_coord+1,:].T + optimaltrajectories[case]['active_muscle_torques'][idx_coord:idx_coord+1,:].T             
    #                 if muscleDrivenJoints[i] in optimaltrajectories[case]['limit_torques_joints']:                        
    #                       idx_coord = optimaltrajectories[case]['limit_torques_joints'].index(muscleDrivenJoints[i])
    #                       ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
    #                             optimaltrajectories[case]['passive_limit_torques'][idx_coord:idx_coord+1,:].T, c=c_col, linestyle='dotted', label='limit torque ' + cases[c])                    
    #                       c_sum += optimaltrajectories[case]['passive_limit_torques'][idx_coord:idx_coord+1,:].T             
    #                 if muscleDrivenJoints[i] in optimaltrajectories[case]['muscle_driven_joints']:                        
    #                       idx_coord = optimaltrajectories[case]['muscle_driven_joints'].index(muscleDrivenJoints[i])
    #                       ax.plot(optimaltrajectories[case]['time'][0,:-1].T,
    #                               c_sum, c=c_col, linestyle='solid', linewidth=3, label='net torque ' + cases[c])
    #             ax.set_title(muscleDrivenJoints[i])
    #             handles, labels = ax.get_legend_handles_labels()
    #             plt.legend(handles, labels, loc='upper right')
    #     plt.setp(axs[-1, :], xlabel='Time (s)')
    #     plt.setp(axs[:, 0], ylabel='(Nm)')
    #     fig.align_ylabels()
    
# %% Process inputs for optimal control problem.   
def processInputsOpenSimAD(baseDir, dataFolder, session_id, trial_name,
                           motion_type, time_window=[], repetition=None,
                           treadmill_speed=0, contact_side='all',
                           overwrite=False, useExpressionGraphFunction=True):
        
    # Path session folder.
    sessionFolder =  os.path.join(dataFolder, session_id)
    
    # Download kinematics and model.    
    pathTrial = os.path.join(sessionFolder, 'OpenSimData', 'Kinematics', 
                             trial_name + '.mot') 
    if not os.path.exists(pathTrial) or overwrite:
        print('Download kinematic data and/or model.')
        _, _ = download_kinematics(session_id, sessionFolder, 
                                   trialNames=[trial_name])
        
    # Get metadata
    metadata = import_metadata(os.path.join(sessionFolder, 'sessionMetadata.yaml'))
    OpenSimModel = metadata['openSimModel']
    
    # TODO: support new shoulder model
    if 'shoulder' in OpenSimModel:
        raise ValueError("""
         The full body model with the ISB shoulder is not yet supported for
         dynamic simulations (https://github.com/stanfordnmbl/opencap-processing/issues/61).
         Consider using the default Full body model instead (LaiUhlrich2022).""")
    
    # Prepare inputs for dynamic simulations.
    # Adjust muscle wrapping.    
    adjust_muscle_wrapping(baseDir, dataFolder, session_id,
                         OpenSimModel=OpenSimModel, overwrite=overwrite, 
                         trial_name=trial_name)
    # Add foot-ground contacts to musculoskeletal model.    
    generate_model_with_contacts(dataFolder, session_id,
                              OpenSimModel=OpenSimModel, overwrite=overwrite,
                              contact_side=contact_side,
                              trial_name=trial_name)
    # Generate external function.    
    generateExternalFunction(baseDir, dataFolder, session_id,
                             OpenSimModel=OpenSimModel,
                             overwrite=overwrite, 
                             treadmill=bool(treadmill_speed),
                             contact_side=contact_side,
                             useExpressionGraphFunction=useExpressionGraphFunction,
                             trial_name=trial_name)
    
    # Get settings.
    settings = get_setup(motion_type)
    # Add time to settings if not specified.
    pathMotionFile = os.path.join(sessionFolder, 'OpenSimData', 'Kinematics',
                                  trial_name + '.mot')
    if (repetition is not None and 
        ('squats' in motion_type or 'sit_to_stand' in motion_type)): 
        if 'squats' in motion_type:           
            sessionDir = os.path.join(dataFolder, session_id)
            squat = squat_analysis(
                sessionDir, trial_name,
                lowpass_cutoff_frequency_for_coordinate_values=4,
                n_repetitions=-1)
            squat_events = squat.get_squat_events()            
            times_window = [list(row) for row in squat_events['eventTimes']]
        elif 'sit_to_stand' in motion_type:
            _, _, times_window = segment_STS(pathMotionFile, visualize=True)
        time_window = times_window[repetition]
        time_window = [x.item() for x in time_window]
        settings['repetition'] = repetition
    else:
        motion_file = storage_to_numpy(pathMotionFile)
        # If no time window is specified, use the whole motion file.
        if not time_window:            
            time_window = [motion_file['time'][0], motion_file['time'][-1]]
        # If -1 is specified for start or end time, use the motion start or end time.
        if time_window[0] == -1:
            time_window[0] = motion_file['time'][0]
        if time_window[1] == -1:
            time_window[1] = motion_file['time'][-1]
        # If time window is specified outside the motion file, use the motion file start or end time.
        if time_window[0] < motion_file['time'][0]:
            time_window[0] = motion_file['time'][0]
        if time_window[1] > motion_file['time'][-1]:
            time_window[1] = motion_file['time'][-1]
            
    settings['timeInterval'] = time_window
    
    # Get demographics.    
    settings['mass_kg'] = metadata['mass_kg']
    settings['height_m'] = metadata['height_m']
    
    # Treadmill speed.
    settings['treadmill_speed'] = treadmill_speed
    
    # Trial name
    settings['trial_name'] = trial_name
    
    # OpenSim model name
    settings['OpenSimModel'] = OpenSimModel

    # Whether to use the expression graph function or (old) external function.
    settings['useExpressionGraphFunction'] = useExpressionGraphFunction

    # Contact side
    settings['contact_side'] = contact_side
    
    return settings

# %% Adjust dummy_motion for polynomial fitting.

def adjustBoundsAndDummyMotion(polynomial_bounds, updated_bounds, pathDummyMotion, pathModelFolder, trialName,
                               overwriteDummyMotion=False):
    # Modify the values of polynomial_bounds based on the values in
    # updated_bounds. 
    for u_b in updated_bounds:
        for c_m in updated_bounds[u_b]:
            polynomial_bounds[u_b][c_m] = updated_bounds[u_b][c_m]
            
    pathAdjustedDummyMotion = os.path.join(pathModelFolder, 'dummy_motion_' + trialName + '.mot')    
    # Generate dummy motion if not exists or if overwrite is True.
    if not os.path.exists(pathAdjustedDummyMotion) or overwriteDummyMotion:
        print('We are adjusting the ROM used for polynomial fitting, but please make sure that the motion to track looks realistic')
        table = opensim.TimeSeriesTable(pathDummyMotion)
        coordinates_table_jointset = list(table.getColumnLabels())
        coordinates_table = [c.split('/')[3] for c in coordinates_table_jointset]
        data = table.getMatrix().to_numpy()
        for u_b in updated_bounds:
            idx_u_b = coordinates_table.index(u_b)
            data[:,idx_u_b] = (polynomial_bounds[u_b]["max"]-polynomial_bounds[u_b]["min"])*np.random.uniform(0.0,1.0,data.shape[0]) + polynomial_bounds[u_b]["min"]
            
        labels = ['time'] + coordinates_table_jointset
        t_dummy_motion = np.array(table.getIndependentColumn())
        t_dummy_motion = np.expand_dims(t_dummy_motion, axis=1)
        data = np.concatenate((t_dummy_motion, data),axis=1)
        
        numpy_to_storage(labels, data, pathAdjustedDummyMotion, datatype='IK')
    
    return polynomial_bounds, pathAdjustedDummyMotion
