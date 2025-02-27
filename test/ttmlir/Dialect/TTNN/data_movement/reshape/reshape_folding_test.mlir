// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s
// Tests if we fold when translating from "ttir.reshape" which is called on the two same shapes.
module @reshape_test {
  func.func @main(%arg0: tensor<1xi32>) -> (tensor<1xi32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<1xi32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32]}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK-NOT: %[[C:.*]] = "ttnn.reshape"[C:.*]]
    // CHECK: return %arg0 : tensor<1xui32, #{{.*}}>
    return %1 : tensor<1xi32>
  }
}
