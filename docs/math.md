# Mathematical invariants

At most two selected points can occur in any row or column. A 150-point configuration on a 75 by 75 grid must therefore have exactly two points in every row and exactly two in every column.

The incidence matrix is consequently a 2-regular bipartite graph between rows and columns. Every even cycle can be edge-coloured alternately, so the selected edges decompose into two perfect matchings, represented by permutations `P` and `Q`, with `P[r] != Q[r]` for every row.

All collinearity decisions use integer determinants. Lines are canonically represented as primitive integer triples `(A,B,C)` for `A*x + B*y + C = 0`, with a fixed sign convention.
