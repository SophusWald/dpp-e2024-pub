-- Computation based on reduce min: performance
-- ==
-- entry: mapomin_primal mapomin_vjp2 mapomin_manual 
-- compiled random input { [10000000]f32 f32 }
-- compiled random input {[100000000]f32 f32 }

def primal [n] (as: [n]f32) : f32 = 
  let bs = map (\a -> let b = a*a - 0.5*a in b) as
  let y  = reduce_comm (f32.min) f32.highest bs
  in  y

entry mapomin_primal [n] (as : [n]f32) (_adj1 : f32) : f32 =
  primal as

entry mapomin_vjp2 [n] (as : [n]f32) (adj : f32) : (f32, [n]f32) =
  vjp2 primal as adj

----------------------------------------------------------------
--
-- Task 1: 
--   Please implement the function below such that it implements
--     the manual-application of reverse-mode AD to the
--     `primal` function above (which is a composition of
--     a map and a reduce with min as operator)
--   Meaning: the semanics of this function should be equivalent
--       with `vjp2 primal as y_bar` but it should not use
--       `vjp` or `vjp2` or `jvp` or `jvp2` at all.
--   See slides entitled "Classical API for AD", "Special Case: Min/Max" 
--     and "Differentiating Map: The Simple Case" in L7and8-AD.pdf
--   Essentially, you will need to follow the rules and write the
--     code of the primal trace (remember to lift the min operator)
--     followed by the code of the reverse trace, which first computes
--     the adjoint of the `reduce min` and then the adjoint of the map.
--   Function Arguments:
--     `as` is the input array for `primal`
--     `y_bar` is the adjoint of the result fo primal.
--   Function Result:
--     the result of the primal (original program)
--     the adjoint of `as`
----------------------------------------------------------------
def min_lift ((a,b) : (i64,f32)) ((c,d) : (i64,f32)): (i64,f32) = if b < d then (a,b) else if b > d then (c,d) else (i64.min a c,b)
def task_mapomin [n] (as : [n]f32) (y_bar : f32) : (f32, [n]f32) =
  -- First we find the index that will be increased
  let bs = map (\a -> let b = a*a - 0.5*a in b) as
  let (i_max,y) = zip (iota n) bs |> reduce_comm min_lift (n,f32.highest)
  -- Secondly we calculate the adjoint for that input.
  let dev = (2 * as[i_max] - 0.5) * y_bar
  in (y, scatter (replicate n 0f32) [i_max] [dev])

entry mapomin_manual [n] (inp : [n]f32) (adj : f32) : (f32, [n]f32) =
  task_mapomin inp adj

-- Computation based on reduce min: validation
-- ==
-- entry: validate
-- compiled input { [5f32, 4f32, 3f32, 2f32, 1f32, 5f32] 1.1f32 } output {true}
-- compiled random input { [10000000]f32 f32 } output {true}
-- compiled random input {[100000000]f32 f32 } output {true}

entry validate [n] (inp : [n]f32) (adj : f32) : bool =
  let (x_vjp, adj_vjp) = vjp2  primal  inp adj
  let (x_man, adj_man) = task_mapomin inp adj
  in  x_vjp == x_man &&
      ( map2 (\x y -> 0.0000001 >= f32.abs (x - y) ) adj_vjp adj_man
        |> reduce (&&) true )
