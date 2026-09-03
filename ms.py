import argparse

import torch

import pyrxmesh as rx
import massspring


parser = argparse.ArgumentParser()
parser.add_argument("mesh")
parser.add_argument("--weight", type=float, default=1.0)
args = parser.parse_args()

rx.init(0)
mesh = rx.RXMeshStatic(args.mesh)
energy = massspring.make_energy(mesh, {"weight": args.weight})
x = torch.as_tensor(
    mesh.vertices(), dtype=torch.float32, device="cuda"
).requires_grad_()
loss = energy.torch(x)
loss.backward()
print("loss:", loss.item())
print("gradient shape:", tuple(x.grad.shape))
