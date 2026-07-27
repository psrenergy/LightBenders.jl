module TestNewsvendorRegularized

using Test
using LightBenders
using JuMP
using HiGHS

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
    # state variable
    if stage == 1
        @variable(sp, 0 <= bought <= inputs.max_storage)
        LightBenders.set_first_stage_state(sp, :bought, bought)
    elseif stage == 2
        @variable(sp, bought)
        LightBenders.set_second_stage_state(sp, :bought, bought)
    end
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

# Scenario-indexed builder used with rebuild_second_stage_per_scenario: the demand
# is baked into the model, so the modifier is a no-op.
function second_stage_builder_per_scenario(sp, inputs, s)
    bought = sp[:bought]

    @variable(sp, sold >= 0)
    @variable(sp, returned >= 0)
    @constraint(sp, sold_dem_con, sold <= inputs.demand[s])
    @constraint(sp, balance, sold + returned <= bought)
    @objective(sp, Min, -sold * inputs.sell_price - returned * inputs.return_price)
    return sp
end

function second_stage_modifier_noop(sp, inputs, s)
    return nothing
end

const LEVEL_SET_HIGHS_ATTRIBUTES = Pair{String, Any}[
    "solver" => "ipm",
    "run_crossover" => "off",
]

function newsvendor_benders(;
    cut_strategy = LightBenders.CutStrategy.MultiCut,
    risk_measure = LightBenders.RiskNeutral(),
    regularization = LightBenders.NoRegularization(),
    rebuild_second_stage_per_scenario = false,
    verbose = false,
)
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
        cut_strategy = cut_strategy,
        risk_measure = risk_measure,
        regularization = regularization,
        rebuild_second_stage_per_scenario = rebuild_second_stage_per_scenario,
        verbose = verbose,
    )

    builder = if rebuild_second_stage_per_scenario
        second_stage_builder_per_scenario
    else
        second_stage_builder
    end
    modifier = if rebuild_second_stage_per_scenario
        second_stage_modifier_noop
    else
        second_stage_modifier
    end

    policy = LightBenders.train(;
        state_variables_builder,
        first_stage_builder,
        second_stage_builder = builder,
        second_stage_modifier = modifier,
        inputs = inputs,
        policy_training_options,
    )

    results = LightBenders.simulate(;
        state_variables_builder,
        first_stage_builder,
        second_stage_builder = builder,
        second_stage_modifier = modifier,
        inputs,
        policy,
        simulation_options = LightBenders.SimulationOptions(
            policy_training_options;
            implementation_strategy = LightBenders.BendersSerialSimulation(),
        ),
    )

    return policy, results
end

function test_level_set_regularization()
    for cut_strategy in
        [LightBenders.CutStrategy.SingleCut, LightBenders.CutStrategy.MultiCut]
        @testset "Regularized Benders Newsvendor $cut_strategy risk neutral" begin
            policy, results = newsvendor_benders(;
                cut_strategy = cut_strategy,
                regularization = LightBenders.LevelSetRegularization(;
                    alpha = 0.5,
                    optimizer_attributes = LEVEL_SET_HIGHS_ATTRIBUTES,
                ),
            )
            @test LightBenders.lower_bound(policy) ≈ -70 atol = 1e-1
            @test LightBenders.upper_bound(policy) ≈ -70 atol = 1e-1
            @test results["objective", 0] ≈ -70 atol = 1e-2
        end
    end
    @testset "Regularized Benders Newsvendor multi cut CVaR" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.MultiCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
            regularization = LightBenders.LevelSetRegularization(;
                alpha = 0.5,
                optimizer_attributes = LEVEL_SET_HIGHS_ATTRIBUTES,
            ),
        )
        @test LightBenders.lower_bound(policy) ≈ -50 atol = 1e-1
        @test LightBenders.upper_bound(policy) ≈ -50 atol = 1e-1
        @test results["objective", 0] ≈ -50 atol = 1e-2
    end
    @testset "Regularized Benders upper bound is non-increasing" begin
        policy, _ = newsvendor_benders(;
            regularization = LightBenders.LevelSetRegularization(;
                alpha = 0.5,
                optimizer_attributes = LEVEL_SET_HIGHS_ATTRIBUTES,
            ),
        )
        UB = policy.progress.UB
        @test all(UB[k+1] <= UB[k] + 1e-9 for k in 1:(length(UB)-1))
    end
    @testset "LevelSetRegularization validates alpha" begin
        @test_throws ArgumentError LightBenders.LevelSetRegularization(alpha = 0.0)
        @test_throws ArgumentError LightBenders.LevelSetRegularization(alpha = 1.0)
    end
end

function test_rebuild_second_stage_per_scenario()
    @testset "Rebuild per scenario Benders Newsvendor" begin
        policy, results = newsvendor_benders(;
            rebuild_second_stage_per_scenario = true,
        )
        @test LightBenders.lower_bound(policy) ≈ -70
        @test LightBenders.upper_bound(policy) ≈ -70
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "Rebuild per scenario regularized Benders Newsvendor" begin
        policy, results = newsvendor_benders(;
            rebuild_second_stage_per_scenario = true,
            regularization = LightBenders.LevelSetRegularization(;
                alpha = 0.5,
                optimizer_attributes = LEVEL_SET_HIGHS_ATTRIBUTES,
            ),
        )
        @test LightBenders.lower_bound(policy) ≈ -70 atol = 1e-1
        @test LightBenders.upper_bound(policy) ≈ -70 atol = 1e-1
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
end

function runtests()
    Base.GC.gc()
    Base.GC.gc()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

TestNewsvendorRegularized.runtests()

end # module TestNewsvendorRegularized
