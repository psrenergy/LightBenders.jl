using LightBenders
using JuMP
using HiGHS

const MOI = JuMP.MOI

Base.@kwdef mutable struct Inputs
    buy_price::Real
    sell_price::Real
    return_price::Real
    max_storage::Int
    demand::Vector{<:Real}
end

function state_variables_builder(inputs, stage)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    sp = LightBenders.SubproblemModel(model)
    if stage == 1
        @variable(sp, 0 <= bought <= inputs.max_storage)
        LightBenders.set_first_stage_state(sp, :bought, bought)
    elseif stage == 2
        @variable(sp, bought)
        LightBenders.set_second_stage_state(sp, :bought, bought)
    end
    return sp
end

function first_stage_builder_max(sp, inputs)
    bought = sp[:bought]
    @constraint(sp, bought <= inputs.max_storage)
    @objective(sp, Max, -bought * inputs.buy_price)
    return sp
end

function second_stage_builder_max(sp, inputs)
    bought = sp[:bought]
    @variable(sp, dem in MOI.Parameter(0.0))
    @variable(sp, sold >= 0)
    @variable(sp, returned >= 0)
    @constraint(sp, sold_dem_con, sold <= dem)
    @constraint(sp, balance, sold + returned <= bought)
    @objective(sp, Max, sold * inputs.sell_price + returned * inputs.return_price)
    return sp
end

function second_stage_modifier(sp, inputs, s)
    dem = sp[:dem]
    JuMP.set_parameter_value(dem, inputs.demand[s])
    return nothing
end

# Test Max Benders
println("="^60)
println("Testing Max Benders Single Cut")
println("="^60)

inputs = Inputs(5, 10, 1, 100, [10, 20, 30])
num_scenarios = length(inputs.demand)

policy_training_options = LightBenders.PolicyTrainingOptions(;
    num_scenarios = num_scenarios,
    lower_bound = -1e6,
    implementation_strategy = LightBenders.SerialTraining(),
    stopping_rule = [LightBenders.GapWithMinimumNumberOfIterations(;
        abstol = 1e-1,
        min_iterations = 2,
    )],
    cut_strategy = LightBenders.CutStrategy.SingleCut,
    verbose = true,
)

policy = LightBenders.train(;
    state_variables_builder,
    first_stage_builder = first_stage_builder_max,
    second_stage_builder = second_stage_builder_max,
    second_stage_modifier,
    inputs = inputs,
    policy_training_options,
)

println("\nTraining Results:")
println("  Lower bound: ", LightBenders.lower_bound(policy))
println("  Upper bound: ", LightBenders.upper_bound(policy))
println("  Expected: 70")

# Test Max Deterministic Equivalent
println("\n")
println("="^60)
println("Testing Max Deterministic Equivalent")
println("="^60)

options = LightBenders.DeterministicEquivalentOptions(; num_scenarios = num_scenarios)

det_eq_results = LightBenders.deterministic_equivalent(;
    state_variables_builder,
    first_stage_builder = first_stage_builder_max,
    second_stage_builder = second_stage_builder_max,
    second_stage_modifier,
    inputs,
    options,
)

println("\nDeterministic Equivalent Results:")
println("  Objective: ", det_eq_results["objective", 0])
println("  Expected: 70")

# Verify results
println("\n")
println("="^60)
println("VERIFICATION")
println("="^60)
lb = LightBenders.lower_bound(policy)
ub = LightBenders.upper_bound(policy)
det_obj = det_eq_results["objective", 0]

if abs(lb - 70) < 1.0 && abs(ub - 70) < 1.0
    println("Benders Max: PASS")
else
    println("Benders Max: FAIL (LB=$lb, UB=$ub, expected 70)")
end

if abs(det_obj - 70) < 1.0
    println("Deterministic Equivalent Max: PASS")
else
    println("Deterministic Equivalent Max: FAIL (obj=$det_obj, expected 70)")
end
