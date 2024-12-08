-- # Tree operations.
--
-- We import a library so we don't have to write a segmented scan
-- ourselves. Remember to run `futhark pkg sync` to download it.
import "lib/github.com/diku-dk/sorts/radix_sort"
import "lib/github.com/diku-dk/segmented/segmented"
-- A traversal is an array of these steps.
type step = #u | #d i32

-- ## Input handling.
--
-- You do not have to modify this. The function 'input.steps' takes as
-- argument a string with steps as discussed in the assignment text,
-- and gives you back an array of type '[]step'.
--
-- Example:
--
-- ```
-- > input.steps "d0 d2 d3 u u d5 u"
-- [#d 0, #d 2, #d 3, #u, #u, #d 5, #u]
-- ```

type char = u8
type string [n] = [n]char

module input: {
  -- | Parse a string into an array of commands.
  val steps [n] : string[n] -> []step
} = {
  def is_space (x: char) = x == ' ' || x == '\n'
  def isnt_space x = !(is_space x)

  def f &&& g = \x -> (f x, g x)

  def dtoi (c: char): i32 = i32.u8 c - '0'

  def is_digit (c: char) = c >= '0' && c <= '9'

  def atoi [n] (s: string[n]): i32 =
    let (sign,s) = if n > 0 && s[0] == '-' then (-1,drop 1 s) else (1,s)
    in sign * (loop (acc,i) = (0,0) while i < length s do
                 if is_digit s[i]
                 then (acc * 10 + dtoi s[i], i+1)
                 else (acc, n)).0

  def to_step (s: []char) : step =
    match s[0]
    case 'u' -> #u
    case _ -> #d (atoi (drop 1 s))

  def steps [n] (s: string[n]) =
    segmented_scan (+) 0 (map is_space s) (map (isnt_space >-> i64.bool) s)
    |> (id &&& rotate 1)
    |> uncurry zip
    |> zip (indices s)
    |> filter (\(i,(x,y)) -> (i == n-1 && x > 0) || x > y)
    |> map (\(i,(x,_)) -> to_step s[i-x+1:i+1])
}

-- ## Task 2.1
-- These three functions matches steps and does something based on that
local def depthchange (x: step) :i64 =
  match x
  case #u -> -1
  case #d i32 -> 1

local def boolchange (x: (i64,step)) :bool =
  match x
  case (i64,#u) -> false
  case (i64,#d i32) -> true
local def change3 (x: (i64,step)) :(i64,i32) =
  match x
  case (i64,#d i32) -> (i64,i32)
  case (i64,#u) -> (i64,0)

def depths (steps: []step) : [](i64,i32) =
  let depthchange = map depthchange steps in
  let alldepths = map2 (-) (scan (+) 0 depthchange) depthchange in
  map change3 (filter boolchange (zip alldepths steps))

-- ## Task 2.2
-- I find an implementation of binary search at https://futhark-lang.org/examples/binary-search.fut
local def binary_search [n] 't (lte: t -> t -> bool) (xs: [n]t) (x: t) : i64 =
  let (l, _) =
    loop (l, r) = (0, n-1) while l < r do
    let t = l + (r - l) / 2
    in if x `lte` xs[t]
       then (l, t)
       else (t+1, r)
  in l
-- This finds the parent of a particular index
local def finder (is: []i64) (i2s: []i64) (d: i64) (y: i64) =
  -- if the depth is 0 or 1, we just make it return 0
  if ((d == 0) || (d == 1)) then 0
  else
    let relevantvalues = is[i2s[d-1]:i2s[d]:1] in
    -- if y's index is higher than all indices of larger depth, it must belong to the last one
    if (y > is[i2s[d]-1]) then is[i2s[d]-1]
    -- otherwise we use binary search to find the appropriate one.
    else
      let index = (binary_search (<=) relevantvalues y) in
      relevantvalues[index-1]

def parents (D: []i64) : []i64 =
 -- D sorted by its values
 let (ds,is) = unzip (radix_sort_by_key (\(x,_) -> i32.i64 x) (i32.num_bits) (i32.get_bit) (zip D (indices D)))
 -- We know want an array with the first index for each depth
 let difs = map2 (-) ds (rotate (-1) ds)
 let (_,i2s) = unzip (filter (\(x,_) -> x != 0) (zip difs (indices difs))) in
 -- with that we use a binary search to find index of the element with right depth and the largest index less than the element we are looking at
 map (\(x,y) -> finder is i2s x y) (zip D (indices D))

-- ## Task 2.3
def subtree_sizes [n] (steps: [n]step) : []i32 =
  -- First we get the depths and the values of the points, and also the parents
  let (ds,start_array) = unzip (depths steps)
  let ps = parents ds
  let max_depth = i64.maximum ds in
  -- We now loop through the depths starting with the largest depth
  loop values = (copy start_array) for i < max_depth do
    -- we find all the values that we need to move up
    let (_,p2s,v2s) = unzip3 (filter (\(x,_,_) -> x == (max_depth - i)) (zip3 ds ps values)) in
    -- we use reduce_by_index to add the immediate children to their parents.
    reduce_by_index values (+) 0 p2s v2s




-- ## Benchmarking
--Lastly we define some functions to help us generate data for our benchmark
def int_to_step [n] (xs: [n]i32) : [n]step =
  map (\x -> #d x) xs
-- For the purpose of parents, i define a variant with same performance, but can work on randomnly generated depth array
-- For the purpose of benchmarking subtree_sizes, i will cheat a bit and instead benchmark it without the first step depths,
-- since we then can generate testing data, and we have already done a test for performance of depths.

local def finder2 (is: []i64) (i2s: []i64) (d: i64) (y: i64) =
  -- if the depth is 0 or 1, we just make it return 0
  if ((d == 0) || (d == 1)) then 0
  else
    let relevantvalues = is[i2s[d-1]:i2s[d]:1] in
    -- if y's index is higher than all indices of larger depth, it must belong to the last one
    if (y > is[i2s[d]-1]) then is[i2s[d]-1]
    -- otherwise we use binary search to find the appropriate one.
    else
      let index = (binary_search (<=) relevantvalues y) in
      relevantvalues[index]


def parents2 (D: []i64) : []i64 =
 -- D sorted by its values
 let (ds,is) = unzip (radix_sort_by_key (\(x,_) -> i32.i64 x) (i32.num_bits) (i32.get_bit) (zip D (indices D)))
 -- We know want an array with the first index for each depth
 let difs = map2 (-) ds (rotate (-1) ds)
 let (_,i2s) = unzip (filter (\(x,_) -> x != 0) (zip difs (indices difs))) in
 -- with that we use a binary search to find index of the element with right depth and the largest index less than the element we are looking at
 map (\(x,y) -> finder2 is i2s x y) (zip D (indices D))


def subtree_sizes_no_depth [n] (ds: [n]i64) (start_array: [n]i32) : []i32 =
  let ps = parents2 ds
  let max_depth = i64.maximum ds in
  -- We now loop through the depths starting with the largest depth
  loop values = (copy start_array) for i < max_depth do
    -- we find all the values that we need to move up
    let (_,p2s,v2s) = unzip3 (filter (\(x,_,_) -> x == (max_depth - i)) (zip3 ds ps values)) in
    -- we use reduce_by_index to add the immediate children to their parents.
    reduce_by_index values (+) 0 p2s v2s


-- Test the depths.
-- ==
-- entry: depth_test
-- random input { [100]i32 } auto output
-- random input { [1000]i32 } auto output
-- random input { [10000]i32 } auto output
-- random input { [100000]i32 } auto output
-- random input { [1000000]i32 } auto output
entry depth_test xs = unzip (depths (int_to_step xs))

-- Test the parents (low depth).
-- ==
-- entry: parents_test
-- random input { [100]i64 2i64} auto output
-- random input { [1000]i64 2i64} auto output
-- random input { [10000]i64 2i64} auto output
-- random input { [100000]i64 2i64} auto output
-- random input { [1000000]i64 2i64} auto output
entry parents_test ds max = parents2 (map (\x -> x % max) ds)

-- Test the subtree_sizes.
-- ==
-- entry: subtree_test
-- random input { [100]i64 [100]i32} auto output
-- random input { [1000]i64 [1000]i32} auto output
-- random input { [10000]i64 [10000]i32} auto output
-- random input { [100000]i64 [100000]i32} auto output
-- random input { [1000000]i64 [1000000]i32} auto output
entry subtree_test ds ys = subtree_sizes_no_depth (map (\x -> x % 100000) ds) ys
