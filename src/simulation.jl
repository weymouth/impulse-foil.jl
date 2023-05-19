using WaterLily,Plots,StaticArrays,CUDA
include("NACA_sdf.jl")
export NACA_sdf

function location(t,t₁=0.1,t₂=0.6,U=1) 
    if t<t₁        # ramp-up
        0.5U/t₁*t^2
    elseif t<t₂    # steady velocity
        U*(t-0.5t₁)
    elseif t<t₂+t₁ # ramp-down
        U*(t-0.5t₁)-0.5U/t₁*(t-t₂)^2
    else           # stop
        U*t₂
    end
end

function foil(;L=32,α=7π/180,t₁=0.1,t₂=1,thk=0.1,Re=5e3,U=1,n=4,m=3,mem=Array)
    # Define (unsteady) pose of foil
    center = L*SA[2,m-1]
    map(x,t) = SA[cos(α) sin(α); -sin(α) cos(α)]*(x-center-L*SA[location(t/L,t₁,t₂),0])

    # Define sdf to symmetric and deflected foil
    sdf(x,t) = L*NACA_sdf(thk)(SA[-x[1],abs(x[2])]/L)

    Simulation((n*L,m*L),(0,0),L;U,ν=U*L/Re,body=AutoBody(sdf,map),mem)
end

begin # create sim and check foil pose
    @assert CUDA.functional()
    sim = foil(L=2^7,mem=CUDA.CuArray,α=10π/180);
    dat = sim.flow.σ[inside(sim.flow.σ)] |> Array;
    @gif for t in 0:0.05:1.5
        measure!(sim,t*sim.L)
        @show t,maximum(sim.flow.V)
        copyto!(dat, sim.flow.σ[inside(sim.flow.σ)]) # copy from GPU
        contour(dat', dpi=300, levels=-1:1, apect_ratio=:equal, legend=false)
    end
end

function WaterLily.CFL(a::Flow)
    @inside a.σ[I] = WaterLily.flux_out(I,a.u)
    min(4.,inv(maximum(a.σ)+5a.ν)) # keep ΔtU/L≤4
end

function get_omega!(dat,sim)
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    copyto!(dat,sim.flow.σ[inside(sim.flow.σ)])
end

plot_vorticity(ω; limit=maximum(abs,ω)) =contourf(ω',dpi=300,
    color=palette(:BuGn), clims=(-limit, limit), linewidth=0,
    aspect_ratio=:equal, legend=false, border=:none)

@gif for t in range(0,6,100) # make a video
    sim_step!(sim,t)
    get_omega!(dat,sim)
    plot_vorticity(dat,limit=4)
    @show t
end