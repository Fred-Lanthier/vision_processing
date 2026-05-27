#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

class FastIK {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double k_null;

    FastIK(const std::string &urdf_path, const std::string &ee_name) {
        // Chargement du modèle Panda
        pinocchio::urdf::buildModel(urdf_path, model);
        data = pinocchio::Data(model);
        
        // Trouver l'index du end-effector (frame)
        if (model.existFrame(ee_name)) {
            ee_frame_id = model.getFrameId(ee_name);
        } else {
            // Fallback sécuritaire
            ee_frame_id = model.frames.size() - 1; 
        }

        // Pré-allocation pour éviter les allocations dynamiques en boucle
        J = pinocchio::Data::Matrix6x(6, model.nv);
        J.setZero();
        v.resize(model.nv);
        v.setZero();

        // Initialisation de la tâche secondaire (null space projection)
        k_null = 0.2;
        w.resize(model.nv);
        w.setZero();
        Jt_temp.resize(model.nv);
        Jt_temp.setZero();
    }

    Eigen::VectorXd get_random_q() {
        return pinocchio::randomConfiguration(model);
    }
    
    int get_nq() const { return model.nq; }
    int get_nv() const { return model.nv; }

    Eigen::Matrix3d projectToSO3(const Eigen::Matrix3d& R_in) const {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_in, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d R = U * V.transpose();
        if (R.determinant() < 0.0) {
            U.col(2) *= -1.0;
            R = U * V.transpose();
        }
        return R;
    }

    // Résolution IK pour une seule pose (Damped Least Squares)
    Eigen::VectorXd solve_single_ik(const Eigen::Matrix4d& target_pose_mat, const Eigen::VectorXd& q_start) {
        Eigen::VectorXd q = q_start;
        const double eps = 1e-5; // Tolérance
        const int IT_MAX = 50;   // Max itérations
        const double damp = 1e-6; // Facteur de damping (DLS)

        if (!target_pose_mat.allFinite() || !q_start.allFinite()) {
            return q_start;
        }

        Eigen::Matrix4d target_clean = target_pose_mat;
        target_clean.block<3,3>(0,0) = projectToSO3(target_clean.block<3,3>(0,0));
        if (!target_clean.allFinite()) {
            return q_start;
        }

        pinocchio::SE3 oMdes(target_clean);

        for (int i=0; i<IT_MAX; i++) {
            // 1. Cinématique directe et placement des frames
            pinocchio::forwardKinematics(model, data, q);
            pinocchio::updateFramePlacement(model, data, ee_frame_id);
            
            // 2. Erreur SE(3) (actuelle - cible)
            const pinocchio::SE3 dMi = oMdes.actInv(data.oMf[ee_frame_id]);
            
            // --- GUARD: Protection contre rotation dégénérée (NaN dans log6) ---
            Eigen::Matrix3d R_check = dMi.rotation();
            double det = R_check.determinant();
            double tr = R_check.trace();
            if (!std::isfinite(det) || std::abs(det - 1.0) > 0.1 || tr < -1.0 || tr > 3.0) {
                break; // Retourner q courant au lieu de crasher
            }
            
            pinocchio::Motion err;
            try {
                err = pinocchio::log6(dMi);
            } catch (...) {
                break;
            }
            
            auto err_vec = err.toVector(); // Stack allocation 6x1
            if(err_vec.norm() < eps) {
                break; // Convergé !
            }
            
            // 3. Jacobienne spatiale (translation + rotation)
            J.setZero();
            pinocchio::computeFrameJacobian(model, data, q, ee_frame_id, pinocchio::LOCAL, J);
            
            // 4. Pseudo-inverse avec Damping (J^T * (J*J^T + damp*I)^-1)
            Eigen::Matrix<double, 6, 6> JJt; // Stack allocation 6x6
            JJt.noalias() = J * J.transpose();
            JJt.diagonal().array() += damp;
            
            // Tâche principale (suivi de la pose cartésienne)
            v.noalias() = -J.transpose() * JJt.ldlt().solve(err_vec);
            
            // Tâche secondaire en espace nul (maintien de la configuration q_start)
            w.noalias() = -k_null * (q - q_start);
            Eigen::Matrix<double, 6, 1> Jw;
            Jw.noalias() = J * w;
            Eigen::Matrix<double, 6, 1> temp = JJt.ldlt().solve(Jw);
            Jt_temp.noalias() = J.transpose() * temp;
            
            v.noalias() += w - Jt_temp;
            
            // 5. Intégration
            double v_norm = v.norm();
            double max_step = 0.2; // Maximum joint increment per iteration (rad)
            if (v_norm > max_step) {
                v *= (max_step / v_norm);
            }
            
            Eigen::VectorXd q_next = pinocchio::integrate(model, q, v);
            if (!q_next.allFinite()) {
                break;
            }
            q = q_next;
        }
        
        pinocchio::forwardKinematics(model, data, q);
        pinocchio::updateFramePlacement(model, data, ee_frame_id);
        return q;
    }

    // Résolution IK par batch (très rapide en Python)
    Eigen::MatrixXd solve_batch(const std::vector<Eigen::Matrix4d>& target_poses, const Eigen::VectorXd& q_start) {
        int N = target_poses.size(); 
        Eigen::MatrixXd results(N, model.nv);
        
        Eigen::VectorXd current_q = q_start;
        
        for(int i=0; i < N; ++i) {
            // Utilise la solution précédente comme point de départ (warm-start)
            current_q = solve_single_ik(target_poses[i], current_q);
            results.row(i) = current_q.transpose();
        }
        return results;
    }

    // Résolution IK et extraction du Jacobien en format 9D (aligné sur le monde)
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> solve_with_jacobian(const Eigen::Matrix4d& target_pose_mat, const Eigen::VectorXd& q_start) {
        Eigen::VectorXd q = solve_single_ik(target_pose_mat, q_start);
        
        // Calcul de la Jacobienne alignée sur le monde à la configuration finale
        pinocchio::Data::Matrix6x J_world(6, model.nv);
        J_world.setZero();
        pinocchio::computeFrameJacobian(model, data, q, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_world);
        
        Eigen::Matrix3d R = data.oMf[ee_frame_id].rotation();
        
        // Conversion de 6D (v, w) en 9D (pos, dr1, dr2)
        Eigen::MatrixXd J9(9, model.nv);
        J9.topRows(3) = J_world.topRows(3);
        J9.middleRows(3, 3) = skew(R.col(0)) * J_world.bottomRows(3);
        J9.bottomRows(3) = skew(R.col(1)) * J_world.bottomRows(3);
        
        return std::make_pair(q, J9);
    }

    // Retourne (Q [N,nv], Js vecteur de matrices [9,nv])
    std::pair<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> solve_batch_with_jacobians(const std::vector<Eigen::Matrix4d>& poses, const Eigen::VectorXd& q_start) {
        int N = poses.size();
        Eigen::MatrixXd Q(N, model.nv);
        std::vector<Eigen::MatrixXd> Js(N);
        
        Eigen::VectorXd current_q = q_start;
        for(int i = 0; i < N; i++) {
            current_q = solve_single_ik(poses[i], current_q);
            Q.row(i) = current_q.transpose();

            // Recalcul avec LOCAL_WORLD_ALIGNED pour cohérence world frame
            pinocchio::computeFrameJacobian(model, data, current_q,
                                            ee_frame_id,
                                            pinocchio::LOCAL_WORLD_ALIGNED, J);

            // Conversion 6D → 9D : [J_pos; skew(r1)·J_ang; skew(r2)·J_ang]
            Eigen::Matrix3d R = data.oMf[ee_frame_id].rotation();
            Eigen::Vector3d r1 = R.col(0);
            Eigen::Vector3d r2 = R.col(1);

            Eigen::MatrixXd J9(9, model.nv);
            J9.topRows(3)      = J.topRows(3);
            J9.middleRows(3,3) = skew(r1) * J.bottomRows(3);
            J9.bottomRows(3)   = skew(r2) * J.bottomRows(3);

            Js[i] = J9;
        }
        return {Q, Js};
    }

private:
    Eigen::Matrix3d skew(const Eigen::Vector3d& v) const {
        Eigen::Matrix3d S;
        S << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return S;
    }
    pinocchio::Model model;
    pinocchio::Data data;
    pinocchio::FrameIndex ee_frame_id;
    pinocchio::Data::Matrix6x J;
    Eigen::VectorXd v;
    Eigen::VectorXd w;
    Eigen::VectorXd Jt_temp;
};

PYBIND11_MODULE(fast_ik_module, m) {
    py::class_<FastIK>(m, "FastIK")
        .def(py::init<const std::string &, const std::string &>())
        .def_readwrite("k_null", &FastIK::k_null)
        .def("get_random_q", &FastIK::get_random_q)
        .def("get_nq", &FastIK::get_nq)
        .def("get_nv", &FastIK::get_nv)
        .def("solve_single_ik", &FastIK::solve_single_ik)
        .def("solve_batch", &FastIK::solve_batch)
        .def("solve_with_jacobian", &FastIK::solve_with_jacobian)
        .def("solve_batch_with_jacobians", &FastIK::solve_batch_with_jacobians);
}
