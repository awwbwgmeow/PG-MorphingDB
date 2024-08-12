#include "utils/vector_tensor.h"
#include "c.h"

extern "C"{

MVec* 
tensor_to_vector(torch::Tensor& tensor)
{
    uint32 dim = tensor.numel();
    uint32 shape_size = tensor.sizes().size();


    MVec* vector = new_mvec(dim, shape_size);

    for (uint32 i=0; i<shape_size; ++i) {
        SET_MVEC_SHAPE_VAL(vector, i, tensor.size(i));
    }

    if(shape_size == 0){
        float value = tensor.item<float>();
        SET_MVEC_VAL(vector, 0, value);
        return vector;
    }

    torch::Tensor flattened_tensor = tensor.view({-1});
    float* data_ptr = flattened_tensor.data_ptr<float>();
    for (int i = 0; i < flattened_tensor.numel(); ++i) {
        SET_MVEC_VAL(vector, i, data_ptr[i]);
    }

    return vector;
}

torch::Tensor 
vector_to_tensor(MVec* vector)
{
    torch::Tensor tensor;
    if(vector->vec_d.shape_size == 1 && vector->vec_d.shape[0] == 0){
        float value = vector->vec_d.data[0];
        tensor = torch::tensor(value);
    }else{
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
        std::vector<int64_t> shape(vector->vec_d.shape, vector->vec_d.shape + vector->vec_d.shape_size);
        tensor = torch::from_blob(vector->vec_d.data, shape, options).clone();
    }
    return tensor;
}

}