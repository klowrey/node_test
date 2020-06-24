# node_test
testing neural ode's

# startup

```bash
$ cd node_test
$ julia --project -O3
```

Then in the REPL, instantiate the project and dependencies, include the script file that's really not organized.

```julia
julia> ]
(node_test) pkg> instantiate
(node_test) pkg> ctrl-c
julia> include("policy.jl")
```

