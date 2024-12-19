-- ==
-- input @ data/kdd_cup.in.gz
-- output @ data/kdd_cup.out
-- random input { 0i32 1024i64 50i32 [10000][256]f32 }

let euclid_dist_2 [d] (pt1: [d]f32) (pt2: [d]f32): f32 =
  f32.sum (map (\x->x*x) (map2 (-) pt1 pt2))

----------------------------------------------------------------
-- Task 4 (a): please implement the function below
-- It denotes the cost function used for solving k-means:
--    (cost points) = f
-- where f is defined in the slide entitled 
--   ``Mathematical Formulation of k-means clustering''
-- of lecture `L7and8-AD.pdf`
----------------------------------------------------------------
let cost [n][k][d] (points: [n][d]f32) (centres: [k][d]f32) : f32 =
  -- This seems like it could be done flattened,
  -- but right now I just the lazy option and made a nested function.
  let dist = map (\x -> reduce_comm f32.min f32.highest (map (\y -> euclid_dist_2 x y) centres )) points
  in reduce (+) 0 dist

let tolerance = 1 : f32

entry main [n][d]
        (_threshold: i32) (k: i64) (max_iterations: i32)
        (points: [n][d]f32) =
  -- Assign arbitrary initial cluster centres.
  let cluster_centres = take k (reverse points)
  let i = 0
  let stop = false
  let (cluster_centres, _i, _stop) =
    loop (cluster_centres : [k][d]f32, i, stop)
    while i < max_iterations && !stop do
      ------------------------------------------------------
      -- Task 4 (b):
      -- please replace the dummy implementation below
      --   that is supposed to compute cost' (Jacobian) and
      --   cost'' (Hessian) of the cost function.
      -- Since `cost` has many inputs and one result,
      --   cost' is to be computed with reverse mode (`vjp`).
      -- Since the Hessian is diagonal, we can compute it
      --   (together with the Jacobian) by applying one `jvp2`
      --   on top of `vjp`, i.e., nesting `vjp` inside `jvp2`
      -- 1. `vjp` is applied to function `(cost points)` in
      --    the point corresponding to the current cluster
      --    center, and with 1 as adjoint (since the result
      --    of `cost` is one float)
      -- 2. `jvp2` is applied to
      --    (a) a lambda that receives as parameter the
      --          cluster centers and applies `vjp` on it
      --    (b) the current cluster centers
      --    (c) a `k x d` matrix of ones (see slides)
      -------------------------------------------------------
      let (cost', cost'') =
        let jacobian clusters = vjp (cost points) x 1
        let hessian clusters = jvp2 jacobian x (unflatten (replicate (k*d) 1))
        in hessian cluster_centres

 -- ( replicate k (replicate d 1f32)
 -- , replicate k (replicate d 1f32))


      --------------------------------------------------------
      -- Task 4 (c):
      -- Please replace the dummy implementation below with
      --   one that correctly computes the new cluster centers/
      -- 
      -- Newton method: x_{k+1} = x_k - f'(x_k) / f''(x_k)
      -- In general f'' is the Hessian and needs matrix
      -- inversion, and then matrix-vector multiplication
      -- with the Jacobian f'. 
      -- In our particular case, the Hessian is diagonal
      -- and is represented as vector cost'' (represented
      -- as a `k x d` matrix), while the Jacobian is vector
      -- cost' (also as a `k x d` matrix).
      -- In this can we can simply use (doubly) vectorized
      -- subtraction and division above to compute the new
      -- centers, i.e., for all 0<=i<k and 0<=j<d:
      --    new_centers_{i,j} = cluster_centers_{i,j} - 
      --                        cost'_{i,j} / cost''_{i,j}
      --------------------------------------------------------
      let new_centres = unflatten (map3 (\x y z -> x - y / z)
                        (flatten cluster_centres) (flatten cost') (flatten cost''))
        -- cluster_centres
      
      -- That's it, do not touch the code below
      -- update stopping condition
      let stop =
        (map2 euclid_dist_2 new_centres cluster_centres |> f32.sum)
        < tolerance
      in (new_centres, i+1, stop)
  in cluster_centres
