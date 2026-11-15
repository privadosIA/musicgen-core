# TorchSharp findings

Behavior of TorchSharp that differs from PyTorch and that every consumer of this core must know about.

## 1. `torch.manual_seed` seeds only the CPU generator

In PyTorch `torch.manual_seed(s)` also seeds every CUDA device. In TorchSharp (0.107) it seeds the CPU generator only.
A generation on a GPU that called just `manual_seed` was not reproducible.

**Evidence** (HeartMuLa.Net): two generations on CUDA with the same seed gave **126 of 128 different tokens** from the first
frame. After also calling `torch.cuda.manual_seed_all(seed)` the same comparison gave **0 of 128** different tokens.

**Rule:** always seed through `Seeding.Apply(seed)`, which calls `manual_seed` and, when CUDA is available, `cuda.manual_seed_all`.
Never call `manual_seed` alone in sampling code or in tests that compare GPU runs.

## 2. Project references outside a solution are built in Debug

`dotnet build Consumer.sln -c Release` builds a `ProjectReference` that is not listed in the solution with the **default configuration (Debug)**, so the
core would silently be unoptimized. The `SetConfiguration` metadata did not change it. Setting `<ShouldUnsetParentConfigurationAndPlatform>false</ShouldUnsetParentConfigurationAndPlatform>`
in the `Directory.Build.props` of **both** the consumer and this repository (the core's own projects reference each other) keeps `Release` as `Release`.
HeartMuLa.Net guards it with a test that compares the `AssemblyConfiguration` of the core and of the test assembly after a solution build.
