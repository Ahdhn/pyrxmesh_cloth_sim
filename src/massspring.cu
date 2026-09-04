#include <pybind11/pybind11.h>

#include <utility>

#include "pyrxmesh/diff_plugin_api.h"

namespace py = pybind11;
using namespace rxmesh;
using namespace pyrxmesh;

using Problem = diff::ScalarGradientProblem<float, 3, VertexHandle>;

void add_terms(Problem& problem, py::dict params)
{
    const float mass                = params["mass"].cast<float>();
    const float half_mass           = 0.5f * mass;
    const float h                   = params["h"].cast<float>();
    const float k                   = params["k"].cast<float>();
    const float neg_mass_times_h_sq = -mass * h * h;
    const float half_k_times_h_sq   = 0.5f * k * h * h;
    const auto  x                   = vertex_attribute<float>(params["x"]);
    const auto  rest_l              = edge_attribute<float>(params["rest_l"]);
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

            const float   rest_squared = rest_l(eh);
            const ActiveT strain = (a - b).squaredNorm() / rest_squared - 1.0f;

            return half_k_times_h_sq * rest_squared * strain * strain;
        });
}

void calc_rest_length(py::object mesh_object, py::dict params)
{
    const auto x      = vertex_attribute<float>(params["x"]);
    auto       rest_l = edge_attribute<float>(params["rest_l"]);

    for_each<Op::EV, 256>(mesh_object,
                          [=] __device__(const EdgeHandle&     eh,
                                         const VertexIterator& iter) mutable {
                              const Eigen::Vector3f a = x.to_eigen<3>(iter[0]);
                              const Eigen::Vector3f b = x.to_eigen<3>(iter[1]);
                              rest_l(eh)              = (a - b).squaredNorm();
                          });
}

void update_velocity(py::object mesh_object, py::dict params)
{
    const float inv_h    = 1.0f / params["h"].cast<float>();
    auto        x        = vertex_attribute<float>(params["x"]);
    const auto  x_tilde  = vertex_attribute<float>(params["x_tilde"]);
    auto        velocity = vertex_attribute<float>(params["velocity"]);

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

    m.def(
        "make_energy",
        [](py::object mesh_object, py::dict params) {
            return diff::make_scalar_energy<float, 3, VertexHandle>(
                mesh_object, std::move(params), add_terms);
        },
        py::arg("mesh"),
        py::arg("params") = py::dict());

    m.def("calc_rest_length",
          &calc_rest_length,
          py::arg("mesh"),
          py::arg("params"));

    m.def("update_velocity",
          &update_velocity,
          py::arg("mesh"),
          py::arg("params"));
}