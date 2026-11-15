using Xunit;

// The libtorch RNG is global: tests that seed it (Seeding.Apply / manual_seed) cannot run in parallel with others that draw from it
// (module constructors such as nn.Linear initialise their weights from it). The whole suite takes well under a second.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
