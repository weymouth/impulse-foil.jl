using WaterLily,Plots,StaticArrays
include("bezier_sdf.jl")
export bezier_sdf

function get_omega!(sim)
    body(I) = sum(WaterLily.ϕ(i,CartesianIndex(I,i),sim.flow.μ₀) for i ∈ 1:2)/2
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * body(I) * sim.L / sim.U
end

plot_vorticity(ω; limit=maximum(abs,ω)) =contourf(ω',dpi=300,
    color=palette(:BuGn), clims=(-limit, limit), linewidth=0,
    aspect_ratio=:equal, legend=false, border=:none)

function foil(L;d=0,Re=38e3,U=1,n=6,m=3)   
    # Pitching motion around the pivot
    ω = 2π*U/L # reduced frequency k=π
    center = SA[1,m/2] # foil placement in domain
    pivot = SA[.1,0] # pitch location from center
    function map(x,t)
        α = 6π/180*cos(ω*t)
        SA[cos(α) sin(α); -sin(α) cos(α)]*(x/L-center-pivot) + pivot
    end

    # Define sdf to symmetric and deflected foil
    symmetric = bezier_sdf(SA[0,0],SA[0,0.1],SA[0.5,0.12],SA[1.,0.])
    deflect(x) = max(0,x-0.3)^2/0.7^2
    sdf(x,time) = L*symmetric(SA[x[1],abs(x[2]+d*deflect(x[1]))])

    Simulation((n*L+2,m*L+2),[U,0],L;ν=U*L/Re,body=AutoBody(sdf,map))
end

function test()
    sim = foil(32,d=0);
    sim_step!(sim,15);
    get_omega!(sim);
    plot_vorticity(sim.flow.σ, limit=10)
end