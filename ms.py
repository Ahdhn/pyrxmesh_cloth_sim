import argparse

import torch

import pyrxmesh as rx
from massspring import calc_rest_length, make_energy, update_velocity


def vertex_mass(mesh, rho, lower, upper):
    extent = upper - lower

    if extent[2] < 1.0e-5:
        area = extent[0] * extent[1]
    elif extent[1] < 1.0e-5:
        area = extent[0] * extent[2]
    elif extent[0] < 1.0e-5:
        area = extent[1] * extent[2]
    else:
        return 1.0

    return float(rho * area / mesh.num_vertices)


def step(
    mesh,
    energy,
    x_attr,
    x_tilde_attr,
    velocity_attr,
    x,
    x_tilde,
    velocity,
    free_mask,
    gradient,
    *,
    time_step,
    tolerance,
    max_iterations,
    history_size,
):
    # Predicted position: x_tilde = x + h * velocity (write directly into the
    # RXMesh x_tilde attribute)
    with torch.no_grad():
        torch.add(x, velocity, alpha=time_step, out=x_tilde)

    # The optimization objective changes after every timestep, so its
    # L-BFGS history is intentionally reset
    optimizer = torch.optim.LBFGS(
        [x_tilde],
        lr=1.0,
        max_iter=max_iterations,
        history_size=history_size,
        tolerance_grad=1.0e-7,
        tolerance_change=tolerance * time_step,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)

        loss = energy.value_and_grad(
            x_tilde,
            out=gradient,
            copy="never",
            gradient_mask=free_mask,
        )

        # The row-major buffer written by RXMesh can be flattened by L-BFGS.
        x_tilde.grad = gradient
        return loss

    loss = optimizer.step(closure)

    update_velocity(
        mesh=mesh,
        time_step=time_step,
        x=x_attr,
        x_tilde=x_tilde_attr,
        velocity=velocity_attr,
    )

    return loss


def simulate(
    *,
    grid_size=16,
    steps=100,
    device_id=0,
    rho=100.0,
    stiffness=4.0e4,
    time_step=0.01,
    tolerance=0.01,
    initial_stretch=1.3,
    lbfgs_max_iter=50,
    lbfgs_history_size=20,
):
    torch.cuda.set_device(device_id)
    rx.init(device_id)

    spacing = 1.0 / float(grid_size - 1)
    vertices, faces = rx.create_plane(
        grid_size,
        grid_size,
        plane=2,
        dx=spacing,
    )
    mesh = rx.RXMeshStatic(vertices, faces)

    lower, upper = mesh.bounding_box()
    mass = vertex_mass(mesh, rho, lower, upper)

    x_attr = mesh.input_vertex_coordinates()
    velocity_attr = mesh.add_vertex_attribute(
        "velocity", dtype="float32", dim=3, location="device")
    rest_l_attr = mesh.add_edge_attribute(
        "rest_l", dtype="float32", dim=1, location="device")
    x_tilde_attr = mesh.add_vertex_attribute(
        "x_tilde", dtype="float32", dim=3, location="device", )

    velocity_attr.reset(0.0, location="device")

    # All three tensors are zero-copy views of RXMesh SoA attributes
    x = x_attr.to_torch("device")
    velocity = velocity_attr.to_torch("device")
    x_tilde = (
        x_tilde_attr.to_torch("device")
        .detach()
        .requires_grad_(True)
    )

    with torch.no_grad():
        # RXMesh pins x < lower_x + numeric_limits<float>::min()
        # True means the vertex is free and its gradient should be retained
        free_mask = x[:, 0] >= (float(lower[0]) + torch.finfo(x.dtype).tiny)

        # Match the RXMesh flag scenario's initial y stretch
        x[:, 1].mul_(initial_stretch)

    # Rest lengths are evaluated after the initial stretch, matching RXMesh.
    calc_rest_length(mesh=mesh, x=x_attr, rest_lengths=rest_l_attr)
    rx.cuda_stream_synchronize()

    energy = make_energy(
        mesh=mesh,
        mass=mass,
        time_step=time_step,
        stiffness=stiffness,
        x=x_attr,
        rest_lengths=rest_l_attr,
    )

    # One persistent row-major gradient allocation for every timestep and
    # every L-BFGS closure evaluation.
    gradient = torch.empty(
        (mesh.num_vertices, 3),
        dtype=x.dtype,
        device=x.device,
    )

    for _ in range(steps):
        step(
            mesh,
            energy,
            x_attr,
            x_tilde_attr,
            velocity_attr,
            x,
            x_tilde,
            velocity,
            free_mask,
            gradient,
            time_step=time_step,
            tolerance=tolerance,
            max_iterations=lbfgs_max_iter,
            history_size=lbfgs_history_size,
        )

    rx.cuda_stream_synchronize()
    return mesh, x_attr, x, velocity, free_mask, gradient


def main():
    parser = argparse.ArgumentParser(
        description="PyRXMesh mass-spring flag simulation"
    )
    parser.add_argument("-n", "--grid-size", type=int, default=16)
    parser.add_argument("-t", "--steps", type=int, default=100)
    parser.add_argument("-d", "--device-id", type=int, default=0)
    parser.add_argument("--rho", type=float, default=100.0)
    parser.add_argument("-k", "--stiffness", type=float, default=4.0e4)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--initial-stretch", type=float, default=1.3)
    parser.add_argument("--lbfgs-max-iter", type=int, default=50)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    args = parser.parse_args()

    _, _, x, velocity, free_mask, _ = simulate(
        grid_size=args.grid_size,
        steps=args.steps,
        device_id=args.device_id,
        rho=args.rho,
        stiffness=args.stiffness,
        time_step=args.time_step,
        tolerance=args.tol,
        initial_stretch=args.initial_stretch,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history_size=args.lbfgs_history_size,
    )

    pinned = ~free_mask
    print(
        f"completed {args.steps} steps; "
        f"max |velocity|={velocity.abs().max().item():.6g}; "
        f"max pinned |velocity|="
        f"{velocity[pinned].abs().max().item():.6g}; "
        f"all finite={bool(torch.isfinite(x).all())}"
    )


if __name__ == "__main__":
    main()
