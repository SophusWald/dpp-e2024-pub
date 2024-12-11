import "trees"

-- Test the parents (low depth).
-- ==
-- entry: parents_test
-- random input { [100]i64 2i64} auto output
-- random input { [1000]i64 2i64} auto output
-- random input { [10000]i64 2i64} auto output
-- random input { [100000]i64 2i64} auto output
-- random input { [1000000]i64 2i64} auto output
-- random input { [10000000]i64 2i64} auto output
entry parents_test ds max = parents2 (map (\x -> x % max) ds)
