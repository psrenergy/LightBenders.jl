abstract type AbstractSimulationImplementation end

struct BendersSerialSimulation <: AbstractSimulationImplementation end

""" 
"""
@enumx SimulationStateHandling begin
    # This is also known as commercial simulation
    StatesFixedInPolicyResult = 0
    StatesRecalculatedInSimulation = 1
end

"""
"""
Base.@kwdef mutable struct SimulationOptions
    # TODO in the near future we should allow users to select the scenarios
    num_scenarios::Int
    implementation_strategy::AbstractSimulationImplementation = BendersSerialSimulation()
    state_handling::SimulationStateHandling.T = SimulationStateHandling.StatesRecalculatedInSimulation
end

function SimulationOptions(policy_training_options::PolicyTrainingOptions; kwargs...)
    return SimulationOptions(;
        num_scenarios = policy_training_options.num_scenarios,
        kwargs...
    )
end

"""
"""
function simulate(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs = nothing,
    policy::Policy,
    simulation_options::SimulationOptions
)
    if simulation_options.implementation_strategy isa BendersSerialSimulation
        return serial_benders_simulate(;
            state_variables_builder,
            first_stage_builder,
            second_stage_builder,
            second_stage_modifier,
            inputs,
            policy,
            simulation_options,
        )
    else
        error("Algorithm not implemented.")
    end
end

