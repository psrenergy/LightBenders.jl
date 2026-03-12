module TestNewsvendorBendersScenarioMap

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

function newsvendor_benders_scenario_map(;
    cut_strategy = LightBenders.CutStrategy.MultiCut,
    risk_measure = LightBenders.RiskNeutral(),
    verbose = true,
)
    # 6 subproblems: demands [10,10,20,20,30,30] grouped into 3 logical scenarios
    inputs = Inputs(5, 10, 1, 100, [10, 10, 20, 20, 30, 30])
    num_scenarios = length(inputs.demand)  # 6
    scenario_map = [1, 1, 2, 2, 3, 3]

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
        scenario_map = scenario_map,
        verbose = verbose,
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

    return policy, results
end

function test_newsvendor_benders_scenario_map()
    @testset "scenario_map single cut risk neutral" begin
        policy, results = newsvendor_benders_scenario_map(;
            cut_strategy = LightBenders.CutStrategy.SingleCut,
            risk_measure = LightBenders.RiskNeutral(),
        )
        @test LightBenders.lower_bound(policy) ≈ -70 atol = 1e-2
        @test LightBenders.upper_bound(policy) ≈ -70 atol = 1e-2
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "scenario_map multi cut risk neutral" begin
        policy, results = newsvendor_benders_scenario_map(;
            cut_strategy = LightBenders.CutStrategy.MultiCut,
            risk_measure = LightBenders.RiskNeutral(),
        )
        @test LightBenders.lower_bound(policy) ≈ -70 atol = 1e-2
        @test LightBenders.upper_bound(policy) ≈ -70 atol = 1e-2
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "scenario_map single cut CVaR" begin
        policy, results = newsvendor_benders_scenario_map(;
            cut_strategy = LightBenders.CutStrategy.SingleCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
        )
        @test LightBenders.lower_bound(policy) ≈ -50 atol = 1e-2
        @test LightBenders.upper_bound(policy) ≈ -50 atol = 1e-2
        @test results["objective", 0] ≈ -50 atol = 1e-2
    end
    @testset "scenario_map multi cut CVaR" begin
        policy, results = newsvendor_benders_scenario_map(;
            cut_strategy = LightBenders.CutStrategy.MultiCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
        )
        @test LightBenders.lower_bound(policy) ≈ -50 atol = 1e-2
        @test LightBenders.upper_bound(policy) ≈ -50 atol = 1e-2
        @test results["objective", 0] ≈ -50 atol = 1e-2
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

TestNewsvendorBendersScenarioMap.runtests()

end # module TestNewsvendorBendersScenarioMap
