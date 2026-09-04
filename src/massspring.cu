#include <pybind11/pybind11.h>

#include <memory>

#include "pyrxmesh/diff_plugin_api.h"

namespace py = pybind11;
using namespace rxmesh;
using namespace pyrxmesh;

using Problem = diff::ScalarGradientProblem<float, 3, VertexHandle>;

void add_terms(Problem&   problem,
               float      mass,
               float      time_step,
               float      stiffness,
               py::handle x_object,
               py::handle rest_lengths_object)
{
    const float half_mass           = 0.5f * mass;
    const float neg_mass_times_h_sq = -mass * time_step * time_step;
    const float half_k_times_h_sq   = 0.5f * stiffness * time_step * time_step;
    const auto  x                   = vertex_attribute<float>(x_object);
    const auto  rest_lengths = edge_attribute<float>(rest_lengths_object);
    const Eigen::Vector3<float> gravity(0.0f, -9.81f, 0.0f);

    // Gravity
    problem.add_term<Op::V>(
        [=] __device__(const auto& vh, auto& opt_var) mutable {
            using ActiveT                   = ACTIVE_TYPE(vh);
            const Eigen::Vector3<ActiveT> q = opt_var.template active<3>(vh);
            return neg_mass_times_h_sq * q.dot(gravity);
        });

    // Inertia
    problem.add_term<Op::V>(
        [=] __device__(const auto& vh, auto& opt_var) mutable {
            using ActiveT                   = ACTIVE_TYPE(vh);
            const Eigen::Vector3<ActiveT> q = opt_var.template active<3>(vh);
            const Eigen::Vector3<float>   position = x.to_eigen<3>(vh);
            return half_mass * (position - q).squaredNorm();
        });

    // Springs
    problem.add_term<Op::EV>(
        [=] __device__(const auto& eh, const auto& iter, auto& opt_var) {
            assert(iter.size() == 2);
            assert(iter[0].is_valid() && iter[1].is_valid());

            using ActiveT = ACTIVE_TYPE(eh);

            const Eigen::Vector3<ActiveT> a =
                opt_var.template active<3>(eh, iter, 0);
            const Eigen::Vector3<ActiveT> b =
                opt_var.template active<3>(eh, iter, 1);

            const float   rest_squared = rest_lengths(eh);
            const ActiveT strain = (a - b).squaredNorm() / rest_squared - 1.0f;

            return half_k_times_h_sq * rest_squared * strain * strain;
        });
}

std::shared_ptr<diff::ScalarEnergyBase> make_energy(
    py::object mesh_object,
    float      mass,
    float      time_step,
    float      stiffness,
    py::object x_object,
    py::object rest_lengths_object)
{
    auto energy = std::make_shared<Problem>(mesh_object);
    add_terms(
        *energy, mass, time_step, stiffness, x_object, rest_lengths_object);
    return energy;
}

void calc_rest_length(py::object mesh_object,
                      py::object x_object,
                      py::object rest_lengths_object)
{
    const auto x            = vertex_attribute<float>(x_object);
    auto       rest_lengths = edge_attribute<float>(rest_lengths_object);

    for_each<Op::EV, 256>(mesh_object,
                          [=] __device__(const EdgeHandle&     eh,
                                         const VertexIterator& iter) mutable {
                              const Eigen::Vector3f a = x.to_eigen<3>(iter[0]);
                              const Eigen::Vector3f b = x.to_eigen<3>(iter[1]);
                              rest_lengths(eh)        = (a - b).squaredNorm();
                          });
}

void update_velocity(py::object mesh_object,
                     float      time_step,
                     py::object x_object,
                     py::object x_tilde_object,
                     py::object velocity_object)
{
    const float inv_h    = 1.0f / time_step;
    auto        x        = vertex_attribute<float>(x_object);
    const auto  x_tilde  = vertex_attribute<float>(x_tilde_object);
    auto        velocity = vertex_attribute<float>(velocity_object);

    for_each_vertex(
        mesh_object, [=] __device__(const VertexHandle& vh) mutable {
            for (int i = 0; i < 3; ++i) {
                velocity(vh, i) = inv_h * (x_tilde(vh, i) - x(vh, i));
                x(vh, i)        = x_tilde(vh, i);
            }
        });
}

PYBIND11_MODULE(_massspring, m)
{
    require_compatible_runtime(m);

    m.def("make_energy",
          &make_energy,
          py::arg("mesh"),
          py::arg("mass"),
          py::arg("time_step"),
          py::arg("stiffness"),
          py::arg("x"),
          py::arg("rest_lengths"));

    m.def("calc_rest_length",
          &calc_rest_length,
          py::arg("mesh"),
          py::arg("x"),
          py::arg("rest_lengths"));

    m.def("update_velocity",
          &update_velocity,
          py::arg("mesh"),
          py::arg("time_step"),
          py::arg("x"),
          py::arg("x_tilde"),
          py::arg("velocity"));
}