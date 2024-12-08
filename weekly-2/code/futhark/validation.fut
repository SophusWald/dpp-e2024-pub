import "trees"

-- Test the depths.
-- ==
-- entry: depth_test
-- input { "d0 d1 d2 u d3 d3 u d4 u d5 u d6 u u d8" } output { [0i64,1i64,2i64,2i64,3i64,3i64,3i64,3i64,2i64] [0i32,1i32,2i32,3i32,3i32,4i32,5i32,6i32,8i32] }
-- input { "d5" } output { [0i64] [5i32] }
-- input { "d5 d5 u d5" } output { [0i64,1i64,1i64] [5i32,5i32,5i32] }
entry depth_test s = unzip (depths (input.steps s))

-- Test the parents.
-- ==
-- entry: parents_test
-- input { [0i64,1i64,2i64,2i64,3i64,3i64,3i64,3i64,2i64,3i64] } output { [0i64,0i64,1i64,1i64,3i64,3i64,3i64,3i64,1i64,8i64] }
-- input { [0i64] } output { [0i64] }
-- input { [0i64,1i64,1i64] } output { [0i64,0i64,0i64] }
entry parents_test ds = parents ds

-- Test the subtree_sizes.
-- ==
-- entry: subtree_test
-- input { "d0 d1 d2 u d3 d3 u d4 u d5 u d6 u u d8" } output { [32i32,32i32,2i32,21i32,3i32,4i32,5i32,6i32,8i32] }
-- input { "d5" } output { [5i32] }
-- input { "d5 d5 u d5" } output { [15i32,5i32,5i32] }
entry subtree_test s = subtree_sizes (input.steps s)
