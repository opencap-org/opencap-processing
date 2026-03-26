'''
    ---------------------------------------------------------------------------
    OpenCap processing: polynomialsOpenSimAD.py
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
    
    This script contains classes and functions to support the approximation of
    muscle-tendon lengths, velocities, and moment arms using polynomial
    approximations of joint positions and velocities.
'''

import numpy as np
import matplotlib.pyplot as plt

class polynomials:
    
    def __init__(self, coefficients, dimension, order):
        
        self.coefficients = coefficients
        self.dimension = dimension
        self.order = order
        
        nq = [0, 0, 0, 0, 0, 0]
        NCoeff = 0
        for nq[0] in range(order + 1):
            if (dimension < 2):
                nq2_s = 0
            else:
                nq2_s = order - nq[0]
            for nq[1] in range(nq2_s + 1):
                if (dimension < 3):
                    nq3_s = 0
                else:
                    nq3_s = order - nq[0] - nq[1]
                for nq[2] in range(nq3_s + 1):
                    if (dimension < 4):
                        nq4_s = 0
                    else:
                        nq4_s = order - nq[0] - nq[1] - nq[2]
                    for nq[3] in range(nq4_s + 1):
                        if (dimension < 5):
                            nq5_s = 0
                        else:
                            nq5_s = order - nq[0] - nq[1] - nq[2] - nq[3]
                        for nq[4] in range(nq5_s + 1):
                            if (dimension < 6):
                                nq6_s = 0
                            else:
                                nq6_s = order - nq[0] - nq[1] - nq[2] - nq[3] - nq[4]
                            for nq[5] in range(nq6_s + 1):
                                NCoeff += 1
        
        if len(coefficients) != NCoeff:
            raise Exception('Expected: {}'.format(NCoeff), 'coefficients', 
                            'but got: {}'.format(len(coefficients)))
            
    def calcValue(self, x):        
        nq = [0, 0, 0, 0, 0, 0]
        coeff_nr = 0
        value = 0
        for nq[0] in range(self.order + 1):
            if (self.dimension < 2):
                nq2_s = 0
            else:
                nq2_s = self.order - nq[0]
            for nq[1] in range(nq2_s + 1):
                if (self.dimension < 3):
                    nq3_s = 0
                else:
                    nq3_s = self.order - nq[0] - nq[1]
                for nq[2] in range(nq3_s + 1):
                    if (self.dimension < 4):
                        nq4_s = 0
                    else:
                        nq4_s = self.order - nq[0] - nq[1] - nq[2]
                    for nq[3] in range(nq4_s + 1):
                        if (self.dimension < 5):
                            nq5_s = 0
                        else:
                            nq5_s = self.order - nq[0] - nq[1] - nq[2] - nq[3]
                        for nq[4] in range(nq5_s + 1):
                            if (self.dimension < 6):
                                nq6_s = 0
                            else:
                                nq6_s = self.order - nq[0] - nq[1] - nq[2] - nq[3] - nq[4]
                            for nq[5] in range(nq6_s + 1):                  
                                valueP = 1
                                for d in range(self.dimension):
                                    valueP *= pow(x[d], nq[d])                            
                                value += valueP * self.coefficients[coeff_nr]
                                coeff_nr += 1
                        
        return value
    
    def calcDerivative(self, x, derivComponent):
        nq = [0, 0, 0, 0, 0, 0]
        coeff_nr = 0
        value = 0
        for nq[0] in range(self.order + 1):
            if (self.dimension < 2):
                nq2_s = 0
            else:
                nq2_s = self.order - nq[0]
            for nq[1] in range(nq2_s + 1):
                if (self.dimension < 3):
                    nq3_s = 0
                else:
                    nq3_s = self.order - nq[0] - nq[1]
                for nq[2] in range(nq3_s + 1):
                    if (self.dimension < 4):
                        nq4_s = 0
                    else:
                        nq4_s = self.order - nq[0] - nq[1] - nq[2]
                    for nq[3] in range(nq4_s + 1):
                        if (self.dimension < 5):
                            nq5_s = 0
                        else:
                            nq5_s = self.order - nq[0] - nq[1] - nq[2] - nq[3]
                        for nq[4] in range(nq5_s + 1):
                            if (self.dimension < 6):
                                nq6_s = 0
                            else:
                                nq6_s = self.order - nq[0] - nq[1] - nq[2] - nq[3] - nq[4]
                            for nq[5] in range(nq6_s + 1):                        
                                if (derivComponent == 0):
                                    nqNonNegative = nq[0] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[0] * pow(x[0], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[d], nq[d])
                                    value += valueP * self.coefficients[coeff_nr]
                                elif (derivComponent == 1):
                                    nqNonNegative = nq[1] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[1] * pow(x[1], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[d], nq[d])
                                    value += valueP * self.coefficients[coeff_nr]
                                elif (derivComponent == 2):
                                    nqNonNegative = nq[2] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[2] * pow(x[2], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[d], nq[d])
                                    value += valueP * self.coefficients[coeff_nr]
                                elif (derivComponent == 3):
                                    nqNonNegative = nq[3] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[3] * pow(x[3], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[d], nq[d])
                                    value += valueP * self.coefficients[coeff_nr]
                                elif (derivComponent == 4):
                                    nqNonNegative = nq[4] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[4] * pow(x[4], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[d], nq[d])
                                    value += valueP * self.coefficients[coeff_nr]
                                elif (derivComponent == 5):
                                    nqNonNegative = nq[5] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[5] * pow(x[5], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[d], nq[d])
                                    value += valueP * self.coefficients[coeff_nr]
                                coeff_nr += 1                 
                                                
        return value
        
class polynomial_estimation:
    
    def __init__(self, dimension, order):
        
        self.dimension = dimension
        self.order = order
        
        nq = [0, 0, 0, 0, 0, 0]
        self.NCoeff = 0
        for nq[0] in range(order + 1):
            if (dimension < 2):
                nq2_s = 0
            else:
                nq2_s = order - nq[0]
            for nq[1] in range(nq2_s + 1):
                if (dimension < 3):
                    nq3_s = 0
                else:
                    nq3_s = order - nq[0] - nq[1]
                for nq[2] in range(nq3_s + 1):
                    if (dimension < 4):
                        nq4_s = 0
                    else:
                        nq4_s = order - nq[0] - nq[1] - nq[2]
                    for nq[3] in range(nq4_s + 1):
                        if (dimension < 5):
                            nq5_s = 0
                        else:
                            nq5_s = order - nq[0] - nq[1] - nq[2] - nq[3]
                        for nq[4] in range(nq5_s + 1):
                            if (dimension < 6):
                                nq6_s = 0
                            else:
                                nq6_s = order - nq[0] - nq[1] - nq[2] - nq[3] - nq[4]
                            for nq[5] in range(nq6_s + 1):
                                self.NCoeff += 1
                    
    def getVariables(self, x):        
        nq = [0, 0, 0, 0, 0, 0]
        coeff_nr = 0
        value = np.zeros((x.shape[0], self.NCoeff))
        for nq[0] in range(self.order + 1):
            if (self.dimension < 2):
                nq2_s = 0
            else:
                nq2_s = self.order - nq[0]
            for nq[1] in range(nq2_s + 1):
                if (self.dimension < 3):
                    nq3_s = 0
                else:
                    nq3_s = self.order - nq[0] - nq[1]
                for nq[2] in range(nq3_s + 1):
                    if (self.dimension < 4):
                        nq4_s = 0
                    else:
                        nq4_s = self.order - nq[0] - nq[1] - nq[2]
                    for nq[3] in range(nq4_s + 1):
                        if (self.dimension < 5):
                            nq5_s = 0
                        else:
                            nq5_s = self.order - nq[0] - nq[1] - nq[2] - nq[3]
                        for nq[4] in range(nq5_s + 1):
                            if (self.dimension < 6):
                                nq6_s = 0
                            else:
                                nq6_s = self.order - nq[0] - nq[1] - nq[2] - nq[3] - nq[4]
                            for nq[5] in range(nq6_s + 1):
                                valueP = 1
                                for d in range(self.dimension):
                                    valueP *= pow(x[:,d], nq[d])                            
                                value[:,coeff_nr ] = valueP
                                coeff_nr += 1
                        
        return value
    
    def getVariableDerivatives(self, x, derivComponent):
        nq = [0, 0, 0, 0, 0, 0]
        coeff_nr = 0
        value = np.zeros((x.shape[0], self.NCoeff))
        for nq[0] in range(self.order + 1):
            if (self.dimension < 2):
                nq2_s = 0
            else:
                nq2_s = self.order - nq[0]
            for nq[1] in range(nq2_s + 1):
                if (self.dimension < 3):
                    nq3_s = 0
                else:
                    nq3_s = self.order - nq[0] - nq[1]
                for nq[2] in range(nq3_s + 1):
                    if (self.dimension < 4):
                        nq4_s = 0
                    else:
                        nq4_s = self.order - nq[0] - nq[1] - nq[2]
                    for nq[3] in range(nq4_s + 1):
                        if (self.dimension < 5):
                            nq5_s = 0
                        else:
                            nq5_s = self.order - nq[0] - nq[1] - nq[2] - nq[3]
                        for nq[4] in range(nq5_s + 1):
                            if (self.dimension < 6):
                                nq6_s = 0
                            else:
                                nq6_s = self.order - nq[0] - nq[1] - nq[2] - nq[3] - nq[4]                        
                            for nq[5] in range(nq6_s + 1):
                                if (derivComponent == 0):
                                    nqNonNegative = nq[0] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[0] * pow(x[:,0], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[:,d], nq[d])
                                    value[:,coeff_nr ] = valueP
                                elif (derivComponent == 1):
                                    nqNonNegative = nq[1] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[1] * pow(x[:,1], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[:,d], nq[d])
                                    value[:,coeff_nr ] = valueP
                                elif (derivComponent == 2):
                                    nqNonNegative = nq[2] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[2] * pow(x[:,2], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[:,d], nq[d])
                                    value[:,coeff_nr ] = valueP
                                elif (derivComponent == 3):
                                    nqNonNegative = nq[3] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[3] * pow(x[:,3], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[:,d], nq[d])
                                    value[:,coeff_nr ] = valueP
                                elif (derivComponent == 4):
                                    nqNonNegative = nq[4] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[4] * pow(x[:,4], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[:,d], nq[d])
                                    value[:,coeff_nr ] = valueP
                                elif (derivComponent == 5):
                                    nqNonNegative = nq[5] - 1
                                    if (nqNonNegative < 0):
                                        nqNonNegative = 0
                                    valueP = nq[5] * pow(x[:,5], nqNonNegative)
                                    for d in range(self.dimension):
                                        if (d == derivComponent):
                                            continue
                                        valueP *= pow(x[:,d], nq[d])
                                    value[:,coeff_nr ] = valueP
                                coeff_nr += 1                 
                                                
        return value
        
def getPolynomialCoefficients(data4PolynomialFitting, joints,
                              muscles, order_min=3, order_max=9,
                              threshold=0.0015,
                              removeBadHipFlexionEntries=True,
                              side='r', debugMode=False):
    
    # Get joint coordinates.   
    idxJoints = [data4PolynomialFitting['coordinate_names'].index(joint) for joint in joints]
    jointCoordinates = data4PolynomialFitting['coordinate_values'][:, idxJoints] * np.pi / 180
    
    # Get muscle-tendon lengths.
    idxMuscles = [data4PolynomialFitting['muscle_names'].index(muscle) for muscle in muscles]
    muscleTendonLengths = data4PolynomialFitting['mtu_lengths'][:, idxMuscles]
    
    # Get moment arms.
    momentArms = data4PolynomialFitting['mtu_moment_arms'][:, idxMuscles, :]
    momentArms = momentArms[:, :, idxJoints]
        
    if removeBadHipFlexionEntries:
        
        # In some cases, the moment arms are bad. This is because of issues
        # with scaling wrapping surfaces. We want to identify those entries 
        # and remove them for the polynomial fitting.

        # Hip flexion     
        idx_hip_flexion = joints.index('hip_flexion_' + side)
        idx_glmax1 = muscles.index('glmax1_' + side)
        idx_glmax2 = muscles.index('glmax2_' + side)
        idx_glmax3 = muscles.index('glmax3_' + side)  
        momentArms_hip_flexion_glmax1 = momentArms[:,idx_glmax1, idx_hip_flexion]    
        idx_bad_hip_flexion_glmax1 = np.where(momentArms_hip_flexion_glmax1 >= -0.02)[0]
        momentArms_hip_flexion_glmax2 = momentArms[:,idx_glmax2, idx_hip_flexion]    
        idx_bad_hip_flexion_glmax2 = np.where(momentArms_hip_flexion_glmax2 >= 0)[0] 
        momentArms_hip_flexion_glmax3 = momentArms[:,idx_glmax3, idx_hip_flexion]    
        idx_bad_hip_flexion_glmax3 = np.where(momentArms_hip_flexion_glmax3 >= 0)[0]        
        idx_iliacus = muscles.index('iliacus_' + side) 
        momentArms_hip_flexion_iliacus = momentArms[:,idx_iliacus, idx_hip_flexion]    
        idx_bad_hip_flexion_iliacus = np.where(momentArms_hip_flexion_iliacus <= 0.0025)[0]   
        idx_bad_hip_flexion = np.concatenate((idx_bad_hip_flexion_glmax1, 
                                              idx_bad_hip_flexion_glmax2,
                                              idx_bad_hip_flexion_glmax3,
                                              idx_bad_hip_flexion_iliacus))
        # Hip adduction
        idx_hip_add = joints.index('hip_adduction_' + side)
        momentArms_hip_add_glmax1 = momentArms[:,idx_glmax1, idx_hip_add]  
        idx_bad_hip_add_glmax1a = np.where(momentArms_hip_add_glmax1 >= -0.005)[0]
        idx_bad_hip_add_glmax1b = np.where(momentArms_hip_add_glmax1 <= -0.07)[0]
        momentArms_hip_add_iliacus = momentArms[:,idx_iliacus, idx_hip_add]    
        idx_bad_hip_add_iliacus = np.where(momentArms_hip_add_iliacus <= -0.02)[0]  
        idx_bad_hip_add = np.concatenate((idx_bad_hip_add_glmax1a,
                                          idx_bad_hip_add_glmax1b,
                                          idx_bad_hip_add_iliacus))
        # Hip rotation
        idx_hip_rot = joints.index('hip_rotation_' + side)
        momentArms_hip_rot_glmax1 = momentArms[:,idx_glmax1, idx_hip_rot]    
        idx_bad_hip_rot_glmax1 = np.where(momentArms_hip_rot_glmax1 >= 0.07)[0]
        
        momentArms_hip_rot_iliacus = momentArms[:,idx_iliacus, idx_hip_rot]    
        idx_bad_hip_rot_iliacus = np.where(momentArms_hip_rot_iliacus <= 0.0025)[0]  
        idx_bad_hip_rot = np.concatenate((idx_bad_hip_rot_glmax1,
                                          idx_bad_hip_rot_iliacus))
        
        idx_bad_hip = np.concatenate((idx_bad_hip_flexion, 
                                      idx_bad_hip_add,
                                      idx_bad_hip_rot))
        
        # Ankle flexion.
        # The edl becomes a plantaflexor in some cases, should not happen.
        idx_ankle_angle = joints.index('ankle_angle_' + side)
        idx_eld = muscles.index('edl_' + side)
        momentArms_ankle_edl = momentArms[:,idx_eld, idx_ankle_angle]  
        idx_bad_ankle_edl = np.where(momentArms_ankle_edl <= 0)[0]
        
        idx_bad = np.concatenate((idx_bad_hip, idx_bad_ankle_edl))
        
        # Remove entries in jointCoordinates, muscleTendonLengths, and momentArms        
        jointCoordinates = np.delete(jointCoordinates, idx_bad, 0)
        muscleTendonLengths = np.delete(muscleTendonLengths, idx_bad, 0)
        momentArms = np.delete(momentArms, idx_bad, 0)
        if debugMode:
            print("{} entries removed for the polynomial fitting because of bad \
    hip and ankle moment arms - mostly because of bad scaling of wrapping surfaces".format(
        idx_bad_hip.shape[0]))
    
    # Detect which muscles actuate which joints.
    momentArms = np.where(np.logical_and(momentArms<=0.003, momentArms>=-0.003), 0, momentArms)
    spanningInfo = np.sum(momentArms, axis=0)
    spanningInfo = np.where(np.logical_and(spanningInfo<=0.01, spanningInfo>=-0.01), 0, 1)
        
    polynomialData = {}
    for i, muscle in enumerate(muscles):
        muscle_momentArms = momentArms[:, i, spanningInfo[i, :]==1]
        muscle_dimension = muscle_momentArms.shape[1]
        muscle_muscleTendonLengths = muscleTendonLengths[:, i]
        
        is_fullfilled = False
        order = order_min
        while not is_fullfilled:
            
            polynomial = polynomial_estimation(muscle_dimension, order)
            mat = polynomial.getVariables(jointCoordinates[:, spanningInfo[i, :]==1])
            
            diff_mat = np.zeros((jointCoordinates.shape[0], mat.shape[1], muscle_dimension))    
            diff_mat_sq = np.zeros((jointCoordinates.shape[0]*(muscle_dimension), mat.shape[1]))  
            for j in range(muscle_dimension):
                diff_mat[:,:,j] = polynomial.getVariableDerivatives(jointCoordinates[:, spanningInfo[i, :]==1], j)
                diff_mat_sq[jointCoordinates.shape[0]*j:jointCoordinates.shape[0]*(j+1),:] = -(diff_mat[:,:,j]).reshape(-1, diff_mat.shape[1])
            
            A = np.concatenate((mat,diff_mat_sq),axis=0)            
            B = np.concatenate((muscle_muscleTendonLengths,(muscle_momentArms.T).flatten()))
            
            # Solve least-square problem.
            coefficients = np.linalg.lstsq(A,B,rcond=None)[0]
            
            # Compute difference with model data.
            # Muscle-tendon lengths.
            muscle_muscleTendonLengths_poly = np.matmul(mat,coefficients)
            muscleTendonLengths_diff_rms = np.sqrt(np.mean(
                    muscle_muscleTendonLengths - muscle_muscleTendonLengths_poly)**2)
            # Moment-arms.
            muscle_momentArms_poly = np.zeros((jointCoordinates.shape[0], muscle_dimension))    
            for j in range(muscle_dimension):        
                muscle_momentArms_poly[:,j] = np.matmul(
                        -(diff_mat[:,:,j]).reshape(-1, diff_mat.shape[1]),coefficients)
                
            momentArms_diff_rms = np.sqrt(np.mean((
                    muscle_momentArms - muscle_momentArms_poly)**2, axis=0))
            
            # Check if criterion is satisfied.
            if (muscleTendonLengths_diff_rms <= threshold and np.max(momentArms_diff_rms) <= threshold):
                is_fullfilled = True
            elif order == order_max:
                is_fullfilled = True
                if debugMode:
                    print("Max order ({}) for {}: rmse_lmte {}, max_rmse_ma {}".format(
                        order_max, muscle, round(muscleTendonLengths_diff_rms, 4),
                        round(np.max(momentArms_diff_rms), 4)))            
            else:
                order += 1
                
        polynomialData[muscle] = {'dimension': muscle_dimension, 'order': order,
                      'coefficients': coefficients, 'spanning': spanningInfo[i, :]}
        
    return polynomialData   

# %% This function plots muscle-tendon lengths and moment arms. Note that this
# is obviously limited to 3D, so muscles actuating more than 2 DOFs will not be
# displayed.
def testPolynomials(data4PolynomialFitting, joints, muscles,
                    f_polynomial, polynomialData, momentArmIndices,
                    trunkMomentArmPolynomialIndices=[]):
    
    # Get joint coordinates.   
    idxJoints = [data4PolynomialFitting['coordinate_names'].index(joint) for joint in joints]
    jointCoordinates = data4PolynomialFitting['coordinate_values'][:, idxJoints] * np.pi / 180
    
    # Get muscle-tendon lengths.
    idxMuscles = [data4PolynomialFitting['muscle_names'].index(muscle) for muscle in muscles]
    muscleTendonLengths = data4PolynomialFitting['mtu_lengths'][:, idxMuscles]
    
    # Get moment arms.
    momentArms = data4PolynomialFitting['mtu_moment_arms'][:, idxMuscles, :]
    momentArms = momentArms[:, :, idxJoints]
    
    # Approximate muscle-tendon lengths
    lMT = np.zeros((len(muscles),muscleTendonLengths.shape[0]))
    dM = np.zeros((len(muscles),len(joints),muscleTendonLengths.shape[0]))
    dM_all = {}
    for k in range(muscleTendonLengths.shape[0]):
        Qsin = jointCoordinates[k, :].T
        Qdotsin = np.zeros((1,Qsin.shape[0]))
        lMT[:,k] = f_polynomial(Qsin, Qdotsin)[0].full().flatten()
        dM[:,:,k] = f_polynomial(Qsin, Qdotsin)[2].full()
        
    for j, joint in enumerate(joints):
        if joint[-1] == 'r' or joint[-1] == 'l':
            dM_all[joint] = dM[momentArmIndices[joint[:-1] + 'l'], j, :]
        else:
            dM_all[joint] = dM[trunkMomentArmPolynomialIndices, j, :]
        
    ny_0 = (np.sqrt(len(muscles))) 
    ny = np.floor(np.sqrt(len(muscles))) 
    ny_a = int(ny)
    ny_b = int(ny)
    if not ny == ny_0:
        ny_b = int(ny+1)
    fig = plt.figure()
    fig.suptitle('Muscle-tendon lengths')
    for i in range(len(muscles)):      
        muscle_obj = muscles[i] #[:-1] + 'r'
        if polynomialData[muscle_obj]['dimension'] == 1:
            temp = polynomialData[muscle_obj]['spanning']==1
            y = (i for i,v in enumerate(temp) if v == True)
            x1 = next(y)
            ax = fig.add_subplot(ny_a, ny_b, i+1)
            ax.scatter(jointCoordinates[:,x1],lMT[i,:])
            ax.scatter(jointCoordinates[:,x1],muscleTendonLengths[:,i],c='r')
            ax.set_title(muscles[i])
            ax.set_xlabel(joints[x1])
        elif polynomialData[muscle_obj]['dimension'] == 2:
            ax = fig.add_subplot(ny_a, ny_b, i+1, projection='3d')
            temp = polynomialData[muscle_obj]['spanning']==1
            y = (i for i,v in enumerate(temp) if v == True)
            x1 = next(y)
            x2 = next(y)
            ax.scatter(jointCoordinates[:,x1],jointCoordinates[:,x2],lMT[i,:])
            ax.scatter(jointCoordinates[:,x1],jointCoordinates[:,x2],muscleTendonLengths[:,i],c='r')
            ax.set_title(muscles[i])
            ax.set_xlabel(joints[x1])
            ax.set_ylabel(joints[x2])   
            
    for i, joint in enumerate(joints):
        fig = plt.figure()
        fig.suptitle('Moment arms: ' + joint)
        NMomentarms = len(momentArmIndices[joint])
        ny_0 = (np.sqrt(NMomentarms)) 
        ny = np.round(ny_0) 
        ny_a = int(ny)
        ny_b = int(ny)
        if (ny == ny_0) == False:
            if ny_a == 1:
                ny_b = NMomentarms
            if ny < ny_0:
                ny_b = int(ny+1)
        for j in range(NMomentarms):
            if joint[-1] == 'r' or joint[-1] == 'l':
                muscle_obj_r = muscles[momentArmIndices[joint[:-1] + 'l'][j]] #[:-1] + 'r'
                muscle_obj = muscles[momentArmIndices[joint[:-1] + 'l'][j]]
            else:
                muscle_obj_r = muscles[trunkMomentArmPolynomialIndices[j]]
                muscle_obj = muscles[trunkMomentArmPolynomialIndices[j]]
            if polynomialData[muscle_obj_r]['dimension'] == 1:
                temp = polynomialData[muscle_obj_r]['spanning']==1
                y = (i for i,v in enumerate(temp) if v == True)
                x1 = next(y)
                ax = fig.add_subplot(ny_a, ny_b, j+1)
                ax.scatter(jointCoordinates[:,x1],dM_all[joint][j,:])
                if joint[-1] == 'r' or joint[-1] == 'l':
                    ax.scatter(jointCoordinates[:,x1],momentArms[:,momentArmIndices[joint[:-1] + 'l'][j],i],c='r')
                else:
                    ax.scatter(jointCoordinates[:,x1],momentArms[:,trunkMomentArmPolynomialIndices[j],i],c='r')
                ax.set_title(muscle_obj)
                ax.set_xlabel(joints[x1])
            if polynomialData[muscle_obj_r]['dimension'] == 2:
                temp = polynomialData[muscle_obj_r]['spanning']==1
                y = (i for i,v in enumerate(temp) if v == True)
                x1 = next(y)
                x2 = next(y)
                ax = fig.add_subplot(ny_a, ny_b, j+1, projection='3d')
                ax.scatter(jointCoordinates[:,x1],jointCoordinates[:,x2],dM_all[joint][j,:])
                if joint[-1] == 'r' or joint[-1] == 'l':
                    ax.scatter(jointCoordinates[:,x1],jointCoordinates[:,x2],momentArms[:,momentArmIndices[joint[:-1] + 'l'][j],i],c='r')
                else:
                    ax.scatter(jointCoordinates[:,x1],jointCoordinates[:,x2],momentArms[:,trunkMomentArmPolynomialIndices[j],i],c='r')
                ax.set_title(muscle_obj)
                ax.set_xlabel(joints[x1])
                ax.set_ylabel(joints[x2])