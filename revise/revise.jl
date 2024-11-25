import Pkg
Pkg.instantiate()

using Revise

Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using LightBenders
@info("""
This session is using LightBenders.jl with Revise.jl.
For more information visit https://timholy.github.io/Revise.jl/stable/.
""")
