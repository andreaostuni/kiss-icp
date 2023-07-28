// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

Sophus::SE3d AlignClouds(const std::vector<Eigen::Vector3d> &source,
                         const std::vector<Eigen::Vector3d> &target,
                         double th) {
    auto compute_jacobian_and_residual = [&](auto i) {
        const Eigen::Vector3d residual = source[i] - target[i];
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(J_r, residual);
    };

    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>{0, source.size()},
        // Identity
        ResultTuple(),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            auto Weight = [&](double residual2) { return square(th) / square(th + residual2); };
            auto &[JTJ_private, JTr_private] = J;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                const double w = Weight(residual.squaredNorm());
                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;
            }
            return J;
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });

    const Eigen::Vector6d x = JTJ.ldlt().solve(-JTr);
    return Sophus::SE3d::exp(x);
}

constexpr int MAX_NUM_ITERATIONS_ = 500;
constexpr double ESTIMATION_THRESHOLD_ = 0.0001;

}  // namespace

namespace kiss_icp {

std::tuple<Sophus::SE3d,Eigen::Matrix6d> RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                           const VoxelHashMap &voxel_map,
                           const Sophus::SE3d &initial_guess,
                           double max_correspondence_distance,
                           double kernel) {
    if (voxel_map.Empty()) return {initial_guess, Eigen::Matrix6d()};

    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        // Equation (10)
        const auto &[src, tgt] = voxel_map.GetCorrespondences(source, max_correspondence_distance);
        // Equation (11)
        auto estimation = AlignClouds(src, tgt, kernel);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (estimation.log().norm() < ESTIMATION_THRESHOLD_) break;
    }
    // Estimate covariance of the measurement
    const auto &[final_src, final_tgt] = voxel_map.GetCorrespondences(source, max_correspondence_distance);
    const double measurement_covariance = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>{0, source.size()},
        // Identity
        0.f,
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, double measure_cov) -> double {
            for (auto i = r.begin(); i < r.end(); ++i) {
                const Eigen::Vector3d residual = final_src[i] - final_tgt[i];
                // TODO: compute normal here (to parallelize)
                measure_cov += residual.squaredNorm();
            }
            return measure_cov / static_cast<double>(source.size());
        },
        // 2nd Lambda: Parallel reduction 
        [&](double a, const double &b) -> double { return a + b; });
    
    // Estimate covariance of the ICP
    // TODO: improve data storages 
    
    // initialize to 10^6
    Eigen::Matrix6d P = Eigen::Matrix6d::Identity() * 1e6;

    for (size_t i = 0; i < frame.size(); ++i) {
        // ComputeNormal(point, );
        const Eigen::Vector3d normal = (frame[i] - final_tgt[i]).normalized();
        // Kalman filter
        // compute point rotation
        const Eigen::Vector3d v = T_icp.so3() * final_tgt[i];

        Eigen::Matrix3d m_r1;
        Eigen::Matrix3d m_r2;
        Eigen::Matrix3d m_r3;
        m_r1 << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0;
        m_r2 << 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
        m_r3 << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        const Eigen::Vector3d d_r1 = m_r1 * v;
        const Eigen::Vector3d d_r2 = m_r2 * v;
        const Eigen::Vector3d d_r3 = m_r2 * v;

        const Eigen::Vector6d H = {normal[0], normal[1], normal[2], (d_r1.transpose() * normal), (d_r2.transpose() * normal), (d_r3.transpose() * normal)};
        const double S = H.transpose() * P * H + measurement_covariance;
        const Eigen::Vector6d K = P * H / S;

        // Update covariance
        P = (Eigen::Matrix6d::Identity() - K * H.transpose()) * P;
    }
    // Spit the final transformation
    return {T_icp * initial_guess, P};
}
}  // namespace kiss_icp
