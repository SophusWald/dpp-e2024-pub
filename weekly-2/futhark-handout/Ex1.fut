import "trees"

-- Test the depths.
-- ==
-- entry: depth_test
-- random input { [100]i32 } auto output
-- random input { [1000]i32 } auto output
-- random input { [10000]i32 } auto output
-- random input { [100000]i32 } auto output
-- random input { [1000000]i32 } auto output
-- random input { [10000000]i32 } auto output
entry depth_test xs = unzip (depths (int_to_step xs))
