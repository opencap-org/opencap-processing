'''
    ---------------------------------------------------------------------------
    OpenCap processing: computeJointLoading.py
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

import opensim
import os
import numpy as np
import copy
import glob

#%% Compute knee adduction moments.
def computeKAM(pathGenericTemplates, outputDir, modelPath, IDPath, IKPath,
               GRFPath, grfType, contactSides, contactSpheres={}, Qds=[]):
    
    print('Computing knee adduction moments.\n')

    jointReactionXmlPath = os.path.join(pathGenericTemplates,'JointReaction',
                                        'Setup_JointReaction.xml')
    statesInDegrees = True # Default input file. If not in header, uses this.
    removeSpheres = True # Delete spheres as force elements
      
    # Load model
    opensim.Logger.setLevelString('error')
    model = opensim.Model(modelPath)
    
    # Remove all actuators and add coordinate actuators.                                         
    forceSet = model.getForceSet()
    i = 0
    while i < forceSet.getSize():
        if removeSpheres:
            forceSet.remove(i)
        else:
            if 'SmoothSphere' not in forceSet.get(i).getConcreteClassName():
                forceSet.remove(i)
            else:
                i+=1
      
    # Coordinates.
    coords = model.getCoordinateSet()
    nCoords = coords.getSize()
    coordNames = [coords.get(i).getName() for i in range(nCoords)]
    
    # Add coordinate actuators.
    actuatorNames = []
    for coord in coords:
        newActuator = opensim.CoordinateActuator(coord.getName())
        newActuator.setName(coord.getName() + '_actuator')
        actuatorNames.append(coord.getName() + '_actuator')
        newActuator.set_min_control(-np.inf)
        newActuator.set_max_control(np.inf)
        newActuator.set_optimal_force(1)
        model.addForce(newActuator)
        
        # Add Prescribed Controllers for Coordinate Actuators.
        #(Needed for joint reaction analysis to work properly).        
        # Construct constant function.
        constFxn = opensim.Constant(0) 
        constFxn.setName(coord.getName() + '_constFxn')         
        # Construct prescribed controller.
        pController = opensim.PrescribedController() 
        pController.setName(coord.getName() + '_controller') 
        pController.addActuator(newActuator)
        # Attach the function to the controller.
        pController.prescribeControlForActuator(0,constFxn)
        model.addController(pController) 
        
    # Get controler set.
    controllerSet = model.getControllerSet()
    
    # Load ID moments
    idTable = opensim.TimeSeriesTable(IDPath)
    idTime = idTable.getIndependentColumn()
    
    # Load kinematic states, compute speeds.
    stateTable = opensim.TimeSeriesTable(IKPath)
    stateNames = stateTable.getColumnLabels()
    stateTime = stateTable.getIndependentColumn()
    try:
        inDegrees = stateTable.getTableMetaDataAsString('inDegrees') == 'yes'
    except:
        inDegrees = statesInDegrees
        print('Using statesInDegrees variable: statesInDegrees is {}'.format(
            statesInDegrees))
    q = np.zeros((len(stateTime),nCoords))
    dt = stateTime[1] - stateTime[0]    
    if len(Qds) > 0:
        qd_t = np.zeros((len(stateTime),nCoords))
        
    for col in stateNames:
        if 'activation' in col:
            stateTable.removeColumn(col)
        else:
            coordCol = coordNames.index(col)
            for t in range(len(stateTime)):
                qTemp = np.asarray(stateTable.getDependentColumn(col)[t])                
                if coords.get(col).getMotionType() == 1 and inDegrees:
                    qTemp = np.deg2rad(qTemp) # convert rotation to rad.
                q[t,coordCol] = copy.deepcopy(qTemp)
            if len(Qds) > 0:
                idx_col = stateNames.index(col)
                qd_t[:,coordCol] = Qds[:, idx_col]
    # Add option to pass q_dot as argument (state of the model), otherwise
    # compute with finite difference.
    if not len(Qds) > 0:
        qd = np.diff(q, axis=0, prepend=np.reshape(q[0,:],(1,nCoords))) / dt
    else:
        qd = qd_t  
        
    # Load and apply GRFs.
    dataSource = opensim.Storage(GRFPath)    
    if grfType == 'sphere':        
        for side in contactSides:
            for c_sphere, sphere in enumerate(contactSpheres[side]):
                newForce = opensim.ExternalForce()
                newForce.setName('{}'.format(sphere))
                appliedToBody = contactSpheres['bodies'][side][c_sphere]
                newForce.set_applied_to_body(appliedToBody)                
                newForce.set_force_expressed_in_body('ground') 
                newForce.set_point_expressed_in_body('ground')
                newForce.set_force_identifier("ground_force_{}_v".format(sphere))
                newForce.set_torque_identifier("ground_torque_{}_".format(sphere))
                newForce.set_point_identifier("ground_force_{}_p".format(sphere))
                newForce.setDataSource(dataSource)
                if removeSpheres:
                    model.addForce(newForce)
                elif i==0:
                    print('GRFs were not applied b/c spheres were used.')
    else:
        raise ValueError("TODO")
    # if grfType == 'sphere':
    #     appliedToBody = ['calcn','calcn','calcn','calcn','toes','toes']
    #     for leg in ['r','l']:
    #         for i in range(len(appliedToBody)):
    #             newForce = opensim.ExternalForce()
    #             newForce.setName('sphere' + leg + str(i+1)) 
    #             newForce.set_applied_to_body(appliedToBody[i] + '_' + leg) 
    #             newForce.set_force_expressed_in_body('ground') 
    #             newForce.set_point_expressed_in_body('ground')
    #             newForce.set_force_identifier(
    #                 'ground_force_s' + str(i+1) + '_' + leg + '_v')
    #             newForce.set_torque_identifier(
    #                 'ground_torque_s' + str(i+1) + '_' + leg + '_')
    #             newForce.set_point_identifier(
    #                 'ground_force_s' + str(i+1) + '_' + leg + '_p')
    #             newForce.setDataSource(dataSource)
    #             if removeSpheres:
    #                 model.addForce(newForce)
    #             elif i==0:
    #                 print('GRFs were not applied b/c spheres were used.')
    # elif grfType == 'sphereResultant':
    #     appliedToBody = ['calcn']
    #     for leg in ['r','l']:
    #         for i in range(len(appliedToBody)):
    #             newForce = opensim.ExternalForce()
    #             newForce.setName('GRF' + '_' + leg + '_' + str(i)) 
    #             newForce.set_applied_to_body(appliedToBody[i] + '_' + 
    #                                          leg.lower()) 
    #             newForce.set_force_expressed_in_body('ground') 
    #             newForce.set_point_expressed_in_body('ground')
    #             newForce.set_force_identifier('ground_force_' + leg + '_v')
    #             newForce.set_torque_identifier('ground_torque_' + leg + '_')
    #             newForce.set_point_identifier('ground_force_' + leg + '_p')
    #             newForce.setDataSource(dataSource)
    #             model.addForce(newForce)
    # elif grfType == 'experimental':
    #     appliedToBody = ['calcn']
    #     for leg in ['R','L']:
    #         for i in range(len(appliedToBody)):
    #             newForce = opensim.ExternalForce()
    #             newForce.setName('GRF' + '_' + leg + '_' + str(i)) 
    #             newForce.set_applied_to_body(appliedToBody[i] + '_' + 
    #                                          leg.lower()) 
    #             newForce.set_force_expressed_in_body('ground') 
    #             newForce.set_point_expressed_in_body('ground')
    #             newForce.set_force_identifier(leg + '_ground_force_v')
    #             newForce.set_torque_identifier(leg + '_ground_torque_')
    #             newForce.set_point_identifier(leg + '_ground_force_p')
    #             newForce.setDataSource(dataSource)
    #             model.addForce(newForce)
    
    # initSystem - done editing model.
    state = model.initSystem()
    
    # Create state Y map.
    yNames = opensim.createStateVariableNamesInSystemOrder(model)
    systemPositionInds = []
    systemVelocityInds = []
    stateNameList = []
    for stateName in coordNames:
        posIdx = np.squeeze(
            np.argwhere([stateName + '/value' in y for y in yNames]))
        velIdx = np.squeeze(
            np.argwhere([stateName + '/speed' in y for y in yNames])) 
        if posIdx.size>0:  
            systemPositionInds.append(posIdx)
            systemVelocityInds.append(velIdx)
            stateNameList.append(stateName)
    
    # Create JRA reporter.
    jointReaction = opensim.JointReaction(jointReactionXmlPath)
    model.addAnalysis(jointReaction) ;
    jointReaction.setModel(model) ;
    jointReaction.printToXML(os.path.join(outputDir, 'JrxnSetup.xml')) ;    
    
    # Loop over time.
    controls = opensim.Vector(nCoords,0) ;
    for iTime in range(len(stateTime)):
        thisTime = stateTime[iTime]    
        if thisTime <= idTime[-1]:
            idRow = idTable.getNearestRowIndexForTime(thisTime)               
            # Set time
            state.setTime(thisTime)                
            # Set state, velocity, actuator controls.
            yVec = np.zeros((state.getNY())).tolist()
            for iCoord, coord in enumerate(coords):
                if '_beta' not in coord.getName():
                    # Loop through states to set values and speeds.
                    yVec[systemPositionInds[iCoord]] = q[iTime,iCoord]
                    yVec[systemVelocityInds[iCoord]] = qd[iTime,iCoord]                    
                    if coord.getMotionType() == 1: # rotation
                        suffix = '_moment'
                    elif coord.getMotionType() == 2: # translation
                        suffix = '_force'                        
                    # Set prescribed controller constant value to control value. 
                    # Controls don't live through joint reaction analysis.
                    thisController = opensim.PrescribedController.safeDownCast(controllerSet.get(coord.getName() + '_controller')) 
                    thisConstFxn = opensim.Constant.safeDownCast(thisController.get_ControlFunctions(0).get(0))
                    thisConstFxn.setValue(idTable.getDependentColumn(coord.getName()+suffix)[idRow])
            # Setting controls this way is redundant, but necessary if want to 
            # do a force reporter in the future.
                    controls.set(iCoord, idTable.getDependentColumn(coord.getName()+suffix)[idRow])
            
            state.setY(opensim.Vector(yVec))
            model.realizeVelocity(state)                
            model.setControls(state,controls)
            
            # Realize acceleration.
            model.realizeAcceleration(state)
            
        # Compute JR.
        if iTime == 0:
            jointReaction.begin(state) 
        else:
            jointReaction.step(state,iTime) 
        if iTime == len(stateTime)-1 or thisTime >=idTime[-1]:
            jointReaction.end(state)
    
    # Finish time loop and output.
    if not removeSpheres:
        grfType = 'spheresUsed_noGRFsApplied'
    outFileBase = 'results_JRA'
    jointReaction.printResults(outFileBase,outputDir,-1,'.sto')
    
    # Load and get KAM.
    outFilePath = glob.glob(
        os.path.join(
            outputDir,
            outFileBase + '_JointReactionAnalysis_ReactionLoads.sto'))[0]    
    thisTable = opensim.TimeSeriesTable(outFilePath)
    results = {} ;
    results['time'] = np.asarray(thisTable.getIndependentColumn())
    nSteps = len(results['time'])
    temp_r = thisTable.getDependentColumn(
        'walker_knee_r_on_tibia_r_in_tibia_r_mx')
    temp_l = thisTable.getDependentColumn(
        'walker_knee_l_on_tibia_l_in_tibia_l_mx')
    
    results['KAM_r'] = np.ndarray((nSteps))
    results['KAM_l'] = np.ndarray((nSteps))
    for i in range(nSteps):
        results['KAM_r'][i] = -temp_r[i]
        results['KAM_l'][i] = temp_l[i]
    
    return results

# %% Compute medial knee contact forces.
def computeMCF(pathGenericTemplates, outputDir, modelPath, activationsPath,
               IKPath, GRFPath, grfType, contactSides, contactSpheres={},
               muscleForceFilePath=None, 
               pathReserveGeneralizedForces=None, Qds=[],pathJRAResults=None, 
               replaceMuscles=False, visualize=False, debugMode=False):
    
    print('Computing medial knee contact forces.\n')
    
    muscForceOverride = muscleForceFilePath!=None
    usingGenForceActuators = pathReserveGeneralizedForces !=None

    if pathJRAResults == None: # If JRA is already computed, skip.
        jointReactionXmlPath = os.path.join(
            pathGenericTemplates,'JointReaction', 'Setup_JointReaction.xml')
        if debugMode:
            print('You supplied muscle forces for computeMCF - these will override forces computed by OpenSim muscle model.')
        
        statesInDegrees = True # Default input file. If not in header, uses this.
        removeSpheres = True # Delete spheres as force elements.
        
        # Load model.
        opensim.Logger.setLevelString('error')
        model = opensim.Model(modelPath)
        
        # Remove spheres.                              
        forceSet = model.getForceSet()        
        i = 0
        while i < forceSet.getSize():
            if 'SmoothSphere' in forceSet.get(i).getConcreteClassName():
                forceSet.remove(i)
            else:
                i+=1
          
        # Coordinates.
        coords = model.getCoordinateSet()
        nCoords = coords.getSize()
        coordNames = [coords.get(i).getName() for i in range(nCoords)]
        
        forceIndsToRemove = []
        for iF,force in enumerate(forceSet):
            if force.getConcreteClassName() == 'CoordinateActuator':
               forceIndsToRemove.append(iF)
               
        forceIndsToRemove.sort(reverse=True)
        for iF in forceIndsToRemove:
            fName = forceSet.get(iF).getName()
            if pathReserveGeneralizedForces != None: 
                forceSet.remove(iF)
                if debugMode:
                    print('deleted ' + fName + ' from force set. Will be added back if a reserve actuator.')
            else:
                if debugMode:
                    print('did not delete ' + fName + ' from force set but it is not actuated.')
            
        # Add coordinate actuators if they are provided in 
        # pathReserveGeneralizedForces.
        actuatorNames = []
        if pathReserveGeneralizedForces !=None:
            idTable = opensim.TimeSeriesTable(pathReserveGeneralizedForces)
            idTime = idTable.getIndependentColumn()
            idLabels = idTable.getColumnLabels()
            
            coordsWithReserves = [[c for c in coordNames if c in f] for f in idLabels]
            coordsWithReserves = [x for x in coordsWithReserves if x != []]
            
            for coordName in coordsWithReserves:
                if isinstance(coordName, list) and len(coordName)==1:
                    coordName = coordName[0]
                newActuator = opensim.CoordinateActuator(coordName)
                newActuator.setName(coordName + '_actuator')
                actuatorNames.append(coordName + '_actuator')
                newActuator.set_min_control(-np.inf)
                newActuator.set_max_control(np.inf)
                newActuator.set_optimal_force(1)
                model.addForce(newActuator)
        
                # Add prescribed controllers for any reserve actuators  .
                # Construct constant function.
                constFxn = opensim.Constant(0) 
                constFxn.setName(coordName + '_constFxn')                 
                # Construct prescribed controller.
                pController = opensim.PrescribedController() 
                pController.setName(coordName + '_controller') 
                pController.addActuator(newActuator) 
                # Attach the function to the controller.
                pController.prescribeControlForActuator(0,constFxn)
                model.addController(pController) 
            
        controllerSet = model.getControllerSet()
        muscles = model.getMuscles()
        
        # Replace muscles.
        if replaceMuscles:
            opensim.DeGrooteFregly2016Muscle.replaceMuscles(model,False)
            for force in forceSet:
                if not force.getConcreteClassName() == 'DeGrooteFregly2016Muscle':
                    continue
                c_force = opensim.DeGrooteFregly2016Muscle.safeDownCast(force)
                # Set to default values or values equivalent to what is used in
                # present code.
                c_force.set_passive_fiber_strain_at_one_norm_force(0.6)
                c_force.set_tendon_strain_at_one_norm_force(0.047359470392808856)

        # Load kinematic states, compute speeds
        stateTable = opensim.TimeSeriesTable(IKPath)
        stateNames = stateTable.getColumnLabels()
        stateTime = stateTable.getIndependentColumn()
        
        # Load activation data - if different than IK for some reason
        actTable = opensim.TimeSeriesTable(IKPath)
        # actNames = actTable.getColumnLabels()
        actTime = actTable.getIndependentColumn()        
        
        # Load force data if provided
        if muscForceOverride:
            overrideForceTable = opensim.TimeSeriesTable(muscleForceFilePath)
            overrideForceTime = overrideForceTable.getIndependentColumn()        
        try:
            inDegrees = stateTable.getTableMetaDataAsString('inDegrees') == 'yes'
        except:
            inDegrees = statesInDegrees
            if debugMode:
                print('Using statesInDegrees variable: statesInDegrees is {}'.format(
                    statesInDegrees))
        q = np.zeros((len(stateTime),nCoords))
        dt = stateTime[1] - stateTime[0]    
        if len(Qds) > 0:
            qd_t = np.zeros((len(stateTime),nCoords))
            
        for col in stateNames:
            if 'activation' in col:
                stateTable.removeColumn(col)
            else:
                coordCol = coordNames.index(col)
                for t in range(len(stateTime)):
                    qTemp = np.asarray(stateTable.getDependentColumn(col)[t])                    
                    if coords.get(col).getMotionType() == 1 and inDegrees:
                        qTemp = np.deg2rad(qTemp) # Convert rotation to rad.
                    q[t,coordCol] = copy.deepcopy(qTemp)
                if len(Qds) > 0:
                    idx_col = stateNames.index(col)
                    qd_t[:,coordCol] = Qds[:, idx_col]
        # Add option to pass q_dot as argument (state of the model), otherwise
        # compute with finite difference.
        if not len(Qds) > 0:
            qd = np.diff(q, axis=0, prepend=np.reshape(q[0,:],(1,nCoords))) / dt
        else:
            qd = qd_t 
        
        # Load activations and muscle forces if overriding.
        muscleNames = []
        act = np.zeros((len(actTime),muscles.getSize()))
        overrideForce = np.copy(act)
        for iMusc,musc in enumerate(muscles):
           muscleNames.append(musc.getName())
           if muscForceOverride:
               forceColName = musc.getName()
               for t in range(len(overrideForceTime)):
                   overrideForce[t,iMusc] = copy.deepcopy(np.asarray(
                                  overrideForceTable.getDependentColumn(forceColName)[t]))
           else:
               actColName = musc.getName()+'/activation'
               for t in range(len(actTime)):
                   act[t,iMusc] = copy.deepcopy(np.asarray(
                                  actTable.getDependentColumn(actColName)[t]))
        
        # Load and apply GRFs.
        dataSource = opensim.Storage(GRFPath)
        if grfType == 'sphere':        
            for side in contactSides:
                for c_sphere, sphere in enumerate(contactSpheres[side]):
                    newForce = opensim.ExternalForce()
                    newForce.setName('{}'.format(sphere))
                    appliedToBody = contactSpheres['bodies'][side][c_sphere]
                    newForce.set_applied_to_body(appliedToBody)                
                    newForce.set_force_expressed_in_body('ground') 
                    newForce.set_point_expressed_in_body('ground')
                    newForce.set_force_identifier("ground_force_{}_v".format(sphere))
                    newForce.set_torque_identifier("ground_torque_{}_".format(sphere))
                    newForce.set_point_identifier("ground_force_{}_p".format(sphere))
                    newForce.setDataSource(dataSource)
                    if removeSpheres:
                        model.addForce(newForce)
                    elif i==0:
                        print('GRFs were not applied b/c spheres were used.')
        else:
            raise ValueError("TODO")
        # if grfType == 'sphere':
        #     appliedToBody = ['calcn','calcn','calcn','calcn','toes','toes']
        #     for leg in ['r','l']:
        #         for i in range(len(appliedToBody)):
        #             newForce = opensim.ExternalForce()
        #             newForce.setName('sphere' + leg + str(i+1)) 
        #             newForce.set_applied_to_body(appliedToBody[i] + '_' + leg) 
        #             newForce.set_force_expressed_in_body('ground') 
        #             newForce.set_point_expressed_in_body('ground')
        #             newForce.set_force_identifier('ground_force_s' + str(i+1) + '_' + leg + '_v')
        #             newForce.set_torque_identifier('ground_torque_s' + str(i+1) + '_' + leg + '_')
        #             newForce.set_point_identifier('ground_force_s' + str(i+1) + '_' + leg + '_p')
        #             newForce.setDataSource(dataSource)
        #             if removeSpheres:
        #                 model.addForce(newForce)
        #             elif i==0:
        #                 print('GRFs were not applied b/c sphere contacts were used.')
        # elif grfType == 'sphereResultant':
        #     appliedToBody = ['calcn']
        #     for leg in ['r','l']:
        #         for i in range(len(appliedToBody)):
        #             newForce = opensim.ExternalForce()
        #             newForce.setName('GRF' + '_' + leg + '_' + str(i)) 
        #             newForce.set_applied_to_body(appliedToBody[i] + '_' + leg.lower()) 
        #             newForce.set_force_expressed_in_body('ground') 
        #             newForce.set_point_expressed_in_body('ground')
        #             newForce.set_force_identifier('ground_force_' + leg + '_v')
        #             newForce.set_torque_identifier('ground_torque_' + leg + '_')
        #             newForce.set_point_identifier('ground_force_' + leg + '_p')
        #             newForce.setDataSource(dataSource)
        #             model.addForce(newForce)
        # elif grfType == 'experimental':
        #     appliedToBody = ['calcn']
        #     for leg in ['R','L']:
        #         for i in range(len(appliedToBody)):
        #             newForce = opensim.ExternalForce()
        #             newForce.setName('GRF' + '_' + leg + '_' + str(i)) 
        #             newForce.set_applied_to_body(appliedToBody[i] + '_' + leg.lower()) 
        #             newForce.set_force_expressed_in_body('ground') 
        #             newForce.set_point_expressed_in_body('ground')
        #             newForce.set_force_identifier(leg + '_ground_force_v')
        #             newForce.set_torque_identifier(leg + '_ground_torque_')
        #             newForce.set_point_identifier(leg + '_ground_force_p')
        #             newForce.setDataSource(dataSource)
        #             model.addForce(newForce)
        
        # initSystem - done editing model.
        state = model.initSystem()
        
        # Create state Y map.
        yNames = opensim.createStateVariableNamesInSystemOrder(model)
        systemPositionInds = []
        systemVelocityInds = []
        stateNameList = []
        for stateName in coordNames:
            posIdx =np.squeeze(np.argwhere([stateName + '/value' in y for y in yNames]))
            velIdx =np.squeeze(np.argwhere([stateName + '/speed' in y for y in yNames])) 
            if posIdx.size>0:  
                systemPositionInds.append(posIdx)
                systemVelocityInds.append(velIdx)
                stateNameList.append(stateName)
                
        systemActivationInds = []
        for muscleName in muscleNames:
            actIdx = np.squeeze(np.argwhere([muscleName + '/activation' in y for y in yNames]))
            systemActivationInds.append(actIdx)
        
        # Create JRA reporter.
        jointReaction = opensim.JointReaction(jointReactionXmlPath)
        model.addAnalysis(jointReaction) ;
        jointReaction.setModel(model) ;
        jointReaction.printToXML(os.path.join(outputDir, 'JrxnSetup.xml')) ;
        
        endTime = []
        if usingGenForceActuators:
            endTime.append(idTime[-1])
        if muscForceOverride:
            endTime.append(overrideForceTable.getIndependentColumn()[-1])
        if len(endTime) == 0:
            endTime = stateTime[-1]
        else:
            endTime = np.max(endTime)

        # Loop over time.
        if usingGenForceActuators:
            controls = opensim.Vector(nCoords,0) ;
        for iTime in range(len(stateTime)):
            thisTime = stateTime[iTime]
        
            if thisTime <= endTime:
                if usingGenForceActuators:
                    idRow = idTable.getNearestRowIndexForTime(thisTime)  
                if muscForceOverride:
                    overrideForceRow = overrideForceTable.getNearestRowIndexForTime(thisTime)
                actRow = actTable.getNearestRowIndexForTime(thisTime)
                
                # Set time
                state.setTime(thisTime)   
                    
                # Set state, velocity, actuator controls
                yVec = np.zeros((state.getNY())).tolist()
                for iCoord, coord in enumerate(coords):
                    if '_beta' not in coord.getName():
                        # Loop through states to set values, speeds, and
                        # activations
                        yVec[systemPositionInds[iCoord]] = q[iTime,iCoord]
                        yVec[systemVelocityInds[iCoord]] = qd[iTime,iCoord]
                        
                        if usingGenForceActuators:
                            actuatorName = [aN for aN in idLabels if coord.getName() in aN]
                            if len(actuatorName) == 1:    
                                # Set prescribed controller constant value to 
                                # control value. Controls don't live through
                                # joint reaction analysis.
                                thisController = opensim.PrescribedController.safeDownCast(controllerSet.get(coord.getName() + '_controller')) 
                                thisConstFxn = opensim.Constant.safeDownCast(thisController.get_ControlFunctions(0).get(0))
                                thisConstFxn.setValue(idTable.getDependentColumn(actuatorName[0])[idRow])
                                
                                # Setting controls this way is redundant, but
                                # necessary if want to use a force reporter
                                # in the future.
                                controls.set(iCoord, idTable.getDependentColumn(actuatorName[0])[idRow])
                
                # Set muscle activations or force.  
                for iMusc,muscName in enumerate(muscleNames):
                    if muscForceOverride:
                        muscles.get(iMusc).overrideActuation(state,True)
                        muscles.get(iMusc).setOverrideActuation(state,overrideForceTable.getDependentColumn(muscName)[overrideForceRow])
                    else:
                       yVec[systemActivationInds[iMusc]] = act[actRow,iMusc]                         
                
                state.setY(opensim.Vector(yVec))
                               
                model.realizeVelocity(state)
                if not muscForceOverride:
                    model.equilibrateMuscles(state) # this changes muscles to appropriate l and v (not 0 from yVec)
                
                if usingGenForceActuators:
                    model.setControls(state,controls)
                
                # Realize acceleration.
                model.realizeAcceleration(state)
                
            # Compute JRA
            if iTime == 0:
                jointReaction.begin(state) 
            else:
                jointReaction.step(state,iTime) 
            if iTime == len(stateTime)-1 or thisTime >= endTime:
                jointReaction.end(state)
        
        # Finish time loop and output.
        if not removeSpheres:
            grfType = 'spheresUsed_noGRFsApplied'
        outFileBase = 'results_JRAforMCF'
        jointReaction.printResults(outFileBase,outputDir,-1,'.sto')        
        
        # Get filename.
        pathJRAResults = glob.glob(os.path.join(outputDir,outFileBase + '_JointReactionAnalysis_ReactionLoads.sto'))[0]
        
    # Load JRA results values and compute MCF.
    thisTable = opensim.TimeSeriesTable(pathJRAResults)
    results = {} ;
    results['time'] = np.asarray(thisTable.getIndependentColumn())
    nSteps = len(results['time'])
    KAM_r = thisTable.getDependentColumn('walker_knee_r_on_tibia_r_in_tibia_r_mx')
    KAM_l = thisTable.getDependentColumn('walker_knee_l_on_tibia_l_in_tibia_l_mx')
    Fy_r = thisTable.getDependentColumn('walker_knee_r_on_tibia_r_in_tibia_r_fy')
    Fy_l = thisTable.getDependentColumn('walker_knee_l_on_tibia_l_in_tibia_l_fy')
    
    d = 0.04 # intercondyler distance (Lerner 2015).    
    results['MCF_r'] = np.ndarray((nSteps))
    results['MCF_l'] = np.ndarray((nSteps))

    # MCF = Fy/2 + KAM/d. Sign changes are due to signs in opensim
    for i in range(nSteps):
        results['MCF_r'][i] = -Fy_r[i]/2 - KAM_r[i]/d
        results['MCF_l'][i] = -Fy_l[i]/2 + KAM_l[i]/d
            
    return results
