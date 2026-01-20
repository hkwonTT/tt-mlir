

#layout = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {
  func.func @spatial_sample() -> (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>) {
    %shared_tensor = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
    %out_tensor0 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
    %out_tensor1 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
    %result0, %result1 = d2m.spatial {grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(0, 0), (1, 1)>, #ttcore.core_range<(2, 0), (3, 1)>]>}
    ins(%shared_tensor : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>)
    outs(%out_tensor0, %out_tensor1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>) {
      ^region0():
        %result_reg_0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%shared_tensor : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>)
        outs(%out_tensor0 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>)  {
          ^compute0(%cb0: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>):
            %8 = d2m.wait %cb0 : <tensor<2x4x!ttcore.tile<32x32, f32>>> -> tensor<2x4x!ttcore.tile<32x32, f32>>
            %9 = d2m.reserve %cb1 : <tensor<2x4x!ttcore.tile<32x32, f32>>> -> tensor<2x4x!ttcore.tile<32x32, f32>>
            %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<2x4x!ttcore.tile<32x32, f32>>) outs(%9 : tensor<2x4x!ttcore.tile<32x32, f32>>) {
              ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
                %11 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                linalg.yield %11 : !ttcore.tile<32x32, f32>
              } -> tensor<2x4x!ttcore.tile<32x32, f32>>
            d2m.yield %10 : (tensor<2x4x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %result_reg_0 : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>)
      },
      {
        ^region1():
          %result_reg_1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
          ins(%shared_tensor : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>)
          outs(%out_tensor1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>)  {
            ^compute0(%cb0: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>>>):
            %8 = d2m.wait %cb0 : <tensor<2x4x!ttcore.tile<32x32, f32>>> -> tensor<2x4x!ttcore.tile<32x32, f32>>
            %9 = d2m.reserve %cb1 : <tensor<2x4x!ttcore.tile<32x32, f32>>> -> tensor<2x4x!ttcore.tile<32x32, f32>>
            %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<2x4x!ttcore.tile<32x32, f32>>) outs(%9 : tensor<2x4x!ttcore.tile<32x32, f32>>) {
              ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
                %11 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                linalg.yield %11 : !ttcore.tile<32x32, f32>
              } -> tensor<2x4x!ttcore.tile<32x32, f32>>
            d2m.yield %10 : (tensor<2x4x!ttcore.tile<32x32, f32>>)
          } : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
          d2m.spatial_yield %result_reg_1 : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>)
      }
     : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
    return %result0, %result1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout>
  }
}
