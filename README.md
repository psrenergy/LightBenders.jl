# LightBenders.jl

[![CI](https://github.com/psrenergy/LightBenders.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/psrenergy/LightBenders.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/psrenergy/LightBenders.jl/graph/badge.svg?token=zfYd247kfX)](https://codecov.io/gh/psrenergy/LightBenders.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Introduction

LightBenders is a Julia package that provides a flexible and efficient implementation of the two-stage Benders decomposition method. Designed for solving large-scale optimization problems, LightBenders is especially suited for problems where decisions are divided into a main (first-stage) problem and a set of scenario-dependent (second-stage) problems.

With support for both serialized and parallel processing of second-stage scenarios, LightBenders allows users to efficiently tackle computationally demanding problems. By leveraging Julia's high-performance capabilities and parallel computing features, the package offers robust performance for solving a wide range of applications, including energy systems, logistics, and supply chain optimization.

### Key Features
- Generic Implementation: Fully customizable for diverse optimization models.
- Scenario Management: Scenarios are handled either sequentially or in parallel, enabling scalability for large instances.
- User-Friendly Interface: Seamlessly integrates with Julia's optimization ecosystem, such as JuMP.
- Parallel Computing Support: Efficiently utilize multiple cores or distributed systems to solve second-stage problems in parallel.

LightBenders is ideal for researchers and practitioners looking to adopt a modular and high-performance approach to Benders decomposition in Julia.

## Getting Started

### Installation

```julia
julia> ] add LightBenders
```

### Example: Newsvendor Model

```julia
using LightBenders
using JuMP
using HiGHS

## definitions
Base.@kwdef mutable struct Inputs
    buy_price::Real
    sell_price::Real
    return_price::Real
    max_storage::Int
    demand::Vector{<:Real}
end

function state_variables_builder(inputs)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    sp = LightBenders.SubproblemModel(model)
    # state variable
    @variable(sp, 0 <= bought <= inputs.max_storage)
    LightBenders.set_state(sp, :bought, bought)

    return sp
end

function first_stage_builder(sp, inputs)
    bought = sp[:bought]

    @constraint(sp, bought <= inputs.max_storage)
    @objective(sp, Min, bought * inputs.buy_price)

    return sp
end

function second_stage_builder(sp, inputs)
    bought = sp[:bought]

    @variable(sp, dem in MOI.Parameter(0.0))
    @variable(sp, sold >= 0)
    @variable(sp, returned >= 0)
    @constraint(sp, sold_dem_con, sold <= dem)
    @constraint(sp, balance, sold + returned <= bought)
    @objective(sp, Min, -sold * inputs.sell_price - returned * inputs.return_price)

    return sp
end

function second_stage_modifier(sp, inputs, s)
    dem = sp[:dem]
    JuMP.set_parameter_value(dem, inputs.demand[s])
    return nothing
end

## call LightBenders
inputs = Inputs(5, 10, 1, 100, [10, 20, 30])
num_scenarios = length(inputs.demand)
policy_training_options = LightBenders.PolicyTrainingOptions(;
    num_scenarios = num_scenarios,
    lower_bound = -1e6,
    implementation_strategy = LightBenders.SerialTraining(),
    stopping_rule = LightBenders.GapWithMinimumNumberOfIterations(;
        abstol = 1e-1,
        min_iterations = 2,
    ),
    cut_strategy = LightBenders.CutStrategy.MultiCut,
)

policy = LightBenders.train(;
    state_variables_builder,
    first_stage_builder,
    second_stage_builder,
    second_stage_modifier,
    inputs = inputs,
    policy_training_options,
)

results = LightBenders.simulate(;
    state_variables_builder,
    first_stage_builder,
    second_stage_builder,
    second_stage_modifier,
    inputs,
    policy,
    simulation_options = LightBenders.SimulationOptions(
        policy_training_options;
        implementation_strategy = LightBenders.BendersSerialSimulation(),
    ),
)
```
