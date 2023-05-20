using Plots,StaticArrays,ForwardDiff
"""
    parametric_sdf(curve)

Create a function to find the signed distance from a point `X` in 2D space
to a parametric `curve(t)` where `t ∈ [0,1]`. The resulting function uses 
ForwardDiff and a simple multi-start Newton root-finding method to identifiy 
`t` closest to `X`. The returned function runs on GPU and has optional arguments:

    function(X,tol=0.01,t₀=(0.01,0.5,0.99),itmx=5)

where `tol` is the tolerance in `t`, `t₀` are the multi-start points (along 
with the bounds 0,1), and `itmx` is the max iterations. For example:

    R = 2
    circle_SDF = parametric_sdf(t->SA[R*sin(2π*t),R*cos(2π*t)])
    @assert circle_SDF(SA[3,4],tol=1e-8) ≈ 5-R

Note that sign of the SDF depends on the direction of `curve`. The example above
wraps clockwise such that distances outside the circle are positive.
"""
function parametric_sdf(curve)
    dcurve(t) = ForwardDiff.derivative(curve,t)
    ddcurve(t) = ForwardDiff.derivative(dcurve,t)
    norm(t) = (s=dcurve(t); SA[-s[2],s[1]])
    vector(X,t) = X-curve(t)
    align(X,t) = vector(X,t)'*dcurve(t)
    dalign(X,t) = vector(X,t)'*ddcurve(t)-sum(abs2,dcurve(t))
    distance(X,t) = (v=vector(X,t); copysign(√(v'*v),norm(t)'*v))
    function(X;tol=0.01,t₀=(0.01,0.5,0.99),itmx=5)
        # distance to ends
        dmin = distance(X,0.)
        d = distance(X,1.)
        abs(d)<abs(dmin) && (dmin=d)
        # check for smaller distance along curve
        for t in t₀
            for _ in 1:itmx # Newton root finding
                dt = align(X,t)/dalign(X,t)
                t -= dt
                t = clamp(t,0.,1.)
                (abs(dt)<tol || t==0 || t==1) && break
            end
            d = distance(X,t)
            abs(d)<abs(dmin) && (dmin=d)
        end
        dmin
    end
end

curve(t::T,thk) where T = SA[t^2,T(5thk*(0.2969t-0.126t^2-0.3516t^4+0.2843t^6-0.1036t^8))]
NACA_sdf(thk=0.12) = parametric_sdf(t->curve(t,thk))

NACA_sdf()(SA[-0.5,0])
NACA_sdf()(SA[1.5,0])
NACA_sdf()(SA[.3,0.06+0.5])

function test()
    sdf(x,y) = NACA_sdf()(SA[x,y])
    x = range(-0.5, 2.5, length=100)
    y = range(0, 0.5, length=50)
    z = sdf.(x',y)
    contour(x,y,z,aspect_ratio=:equal, legend=false, border=:none)
    contour!(x,y,z,aspect_ratio=:equal, legend=false, border=:none,levels=[0],color=:green)
    savefig(".\\figures\\distance function.png")
end
