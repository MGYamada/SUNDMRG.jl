"""
string = graphic(sys_block, env_block; sys_label = :l)
visualizes DMRG
"""
function graphic(sys_block, env_block; sys_label = :l)
    str = repeat("=", sys_block.length) * "**" * repeat("-", env_block.length)
    if sys_label == :r
        str = reverse(str)
    elseif sys_label != :l
        throw(ArgumentError("sys_label must be :l or :r"))
    end
    str
end
