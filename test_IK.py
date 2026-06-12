#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pybullet as pb
import rospkg

# =====================================================================
# 🚀 LE SOLVEUR HAUTE VITESSE (PyBullet C++ Engine)
# =====================================================================

if __name__ == "__main__":
    # 1. Démarrer PyBullet en mode DIRECT (Sans GUI, vitesse maximale)
    pb.connect(pb.DIRECT)

    # 2. Charger le robot avec un chemin dynamique ROS
    rospack = rospkg.RosPack()
    vision_processing_path = rospack.get_path('vision_processing')
    urdf_path = vision_processing_path + "/third_party/RDF/collision_avoidance_example/xarm7_urdf/xarm7_FT_EE.urdf"
    
    robot_id = pb.loadURDF(urdf_path, useFixedBase=True)

    # 3. Trouver l'index de l'effecteur (la fourchette / link8)
    num_joints = pb.getNumJoints(robot_id)
    ee_index = -1
    for i in range(num_joints):
        info = pb.getJointInfo(robot_id, i)
        link_name = info[12].decode('utf-8')
        if "link8" in link_name or "hand" in link_name:
            ee_index = i
            break
            
    if ee_index == -1: 
        ee_index = num_joints - 1 # Fallback de sécurité

    movable_joints = [j for j in range(num_joints) if pb.getJointInfo(robot_id, j)[2] != pb.JOINT_FIXED]

    # Pose neutre de départ
    q_current = [-0.000059, -0.125928, 0.000117, -2.193312, -0.000251, 2.064780, 0.785511]
    
    # Centre du cercle et rayon (en mètres)
    center = np.array([0.5, 0.0, 0.4])
    radius = 0.15
    num_points = 16

    # Obtenir l'orientation initiale pour la garder constante
    for j, joint_idx in enumerate(movable_joints[:7]):
        pb.resetJointState(robot_id, joint_idx, q_current[j])
    _, target_orn = pb.getLinkState(robot_id, ee_index)[0:2]

    print(f"🔄 Début du test intensif sur {num_points} poses (Trajectoire circulaire)...")

    times = []
    spatial_errors = []

    # Warm-up (Chargement de la librairie en RAM)
    pb.calculateInverseKinematics(robot_id, ee_index, targetPosition=center.tolist(), targetOrientation=list(target_orn))

    for i in range(num_points):
        # Calcul de la cible sur le cercle
        angle = 2 * np.pi * i / num_points
        target_pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])

        # On place le robot à la dernière position connue
        for j, joint_idx in enumerate(movable_joints[:7]):
            pb.resetJointState(robot_id, joint_idx, q_current[j])

        t_start = time.perf_counter()
        
        # Résolution IK
        q_opt = pb.calculateInverseKinematics(
            robot_id,
            ee_index,
            targetPosition=target_pos.tolist(),
            targetOrientation=list(target_orn),
            maxNumIterations=50,
            residualThreshold=1e-5
        )
        
        t_end = time.perf_counter()
        times.append((t_end - t_start) * 1000)

        # Mise à jour pour la prochaine itération
        q_current = q_opt[:7]

        # Calcul de l'erreur totale
        for j, joint_idx in enumerate(movable_joints[:7]):
            pb.resetJointState(robot_id, joint_idx, q_current[j])
            
        actual_pos, actual_orn = pb.getLinkState(robot_id, ee_index)[0:2]
        err_pos = np.linalg.norm(target_pos - np.array(actual_pos))
        diff_q = pb.getDifferenceQuaternion(target_orn, actual_orn)
        err_rot = 2 * np.arccos(np.clip(diff_q[3], -1.0, 1.0))
        err_tot = np.sqrt(err_pos**2 + err_rot**2)
        spatial_errors.append(err_tot)

    print("="*50)
    print("📊 RÉSULTATS GLOBAUX SUR 1000 POSES CONTINUES (PYBULLET)")
    print("="*50)
    print(f"⏱️ Temps MOYEN par pose  : {np.mean(times):.4f} ms")
    print(f"⏱️ Temps MAX (pire cas)  : {np.max(times):.4f} ms")
    print(f"🎯 Erreur totale moy     : {np.mean(spatial_errors)*1000:.4f} (mm/mrad)")
    print(f"🎯 Erreur totale max     : {np.max(spatial_errors)*1000:.4f} (mm/mrad)")
    print("="*50)

    # =====================================================================
    # ⚡ TEST DE LA LIBRAIRIE C++ PYBIND11 (FAST IK)
    # =====================================================================
    print("\n🔄 Début du test Pybind11 C++ ...")
    
    try:
        from vision_processing import fast_ik_module
        
        # 1. Initialisation de la classe C++
        # On extrait le nom du lien de PyBullet pour le donner à Pinocchio
        ee_name = pb.getJointInfo(robot_id, ee_index)[12].decode('utf-8')
        ik_cpp = fast_ik_module.FastIK(urdf_path, ee_name)
        
        # 2. Préparation des poses 4x4 cibles pour le batch
        targets = []
        rot_matrix = np.array(pb.getMatrixFromQuaternion(target_orn)).reshape(3, 3)
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = pos
            targets.append(T)
        targets_matrix = targets # Liste de matrices 4x4 (std::vector<Eigen::Matrix4d>)
        
        # Point de départ dynamique (s'adapte à n'importe quel URDF grâce à get_nq())
        q_init = np.zeros(ik_cpp.get_nq())
        # On remplit les N premiers joints avec la pose de départ
        n_movable = min(len(q_current), ik_cpp.get_nq())
        q_init[:n_movable] = q_current[:n_movable]
        
        # 3. Exécution Batch
        t_start = time.perf_counter()
        q_results = ik_cpp.solve_batch(targets_matrix, q_init)
        t_end = time.perf_counter()
        
        total_time_ms = (t_end - t_start) * 1000
        mean_time_ms = total_time_ms / num_points

        # Vérification de l'erreur totale avec PyBullet
        spatial_errors_cpp = []
        for i in range(num_points):
            q_res = q_results[i]
            for j, joint_idx in enumerate(movable_joints[:7]):
                pb.resetJointState(robot_id, joint_idx, q_res[j])
            
            actual_pos, actual_orn = pb.getLinkState(robot_id, ee_index)[0:2]
            err_pos = np.linalg.norm(targets[i][:3, 3] - np.array(actual_pos))
            diff_q = pb.getDifferenceQuaternion(target_orn, actual_orn)
            err_rot = 2 * np.arccos(np.clip(diff_q[3], -1.0, 1.0))
            err_tot = np.sqrt(err_pos**2 + err_rot**2)
            spatial_errors_cpp.append(err_tot)

        print("="*50)
        print("📊 RÉSULTATS GLOBAUX SUR 1000 APPELS (PYBIND11 C++)")
        print("="*50)
        print(f"⏱️ Temps MOYEN par pose (Batch IK) : {mean_time_ms:.4f} ms")
        print(f"⏱️ Temps TOTAL pour {num_points} poses   : {total_time_ms:.4f} ms")
        print(f"🎯 Erreur totale moy     : {np.mean(spatial_errors_cpp)*1000:.4f} (mm/mrad)")
        print(f"🎯 Erreur totale max     : {np.max(spatial_errors_cpp)*1000:.4f} (mm/mrad)")
        print("="*50)
        
    except ImportError as e:
        print(f"❌ Impossible d'importer fast_ik_module : {e}")
        print("Assure-toi d'avoir fait `catkin build` et `source devel/setup.bash`")

    pb.disconnect()