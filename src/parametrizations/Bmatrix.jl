# non-constant mass matrix B(x): B(W(s)) is carried by the B graph output
# sys.gB.data.y (just like F(W) is carried by sys.gF.data.y) and enters the
# residue as E2 = [B(W)·DW·f]_k via the product below.

# order-k matrix·vector product Y = B(W)·DWf. row r of the B series is entry
# B[i,j] with (i,j) = rowmap[r], contributing Y[i] += B[i,j]·DWf[j]. the row map
# unifies dense (full column-major map) and sparse (nonzero entries only) B
function bmatvec!(Y::PolynomialArray, gBy::Vector{Polynomial{D}}, DWf::PolynomialArray,
                  rowmap::Vector{Tuple{Int,Int}}, k, ct) where D
    Yt, Dt = Y.tensors, DWf.tensors
    @inbounds for (i1, i2, itgt) in ct[k]
        for r in eachindex(rowmap)
            i, j = rowmap[r]
            Yt[i, itgt] += gBy[r].tensor[i1] * Dt[j, i2]
        end
    end
    return Y
end
