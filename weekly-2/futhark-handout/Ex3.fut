import "trees"

-- Test the subtree_sizes.
-- ==
-- entry: subtree_test
-- random input { [10000]i64 [10000]i32 1i64} auto output
-- random input { [10000]i64 [10000]i32 10i64} auto output
-- random input { [10000]i64 [10000]i32 100i64} auto output
-- random input { [10000]i64 [10000]i32 1000i64} auto output
entry subtree_test ds ys max = subtree_sizes_no_depth (map (\x -> x % max) ds) ys
