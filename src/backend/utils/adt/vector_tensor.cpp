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
        if (tensor.scalar_type() == torch::kFloat32) {
            float value = tensor.item<float>();
            SET_MVEC_VAL(vector, 0, value);
        } else if (tensor.scalar_type() == torch::kInt64) {
            int64_t value = tensor.item<int64_t>();
            SET_MVEC_VAL(vector, 0, static_cast<float>(value));
        }
        return vector;
    }

    torch::Tensor flattened_tensor = tensor.view({-1});
    if (tensor.scalar_type() == torch::kFloat32) {
        float* data_ptr = flattened_tensor.data_ptr<float>();
        for (int i = 0; i < flattened_tensor.numel(); ++i) {
            SET_MVEC_VAL(vector, i, data_ptr[i]);
        }
    } else if (tensor.scalar_type() == torch::kInt64) {
        int64_t* data_ptr = flattened_tensor.data_ptr<int64_t>();
        for (int i = 0; i < flattened_tensor.numel(); ++i) {
            SET_MVEC_VAL(vector, i, static_cast<float>(data_ptr[i]));
        }
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