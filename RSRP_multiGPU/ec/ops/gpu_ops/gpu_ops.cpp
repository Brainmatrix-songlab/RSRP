#include "kernels.h"
#include "pybind11_kernel_helpers.h"

namespace
{
    pybind11::dict MVMRegistrations()
    {
        pybind11::dict dict;
        dict["gpu_BF16"] =
            gpu_ops::EncapsulateFunction(gpu_ops::gpu_BF16);
        return dict;
    }

    PYBIND11_MODULE(gpu_ops, m)
    {
        m.def("get_mvm_registrations", &MVMRegistrations);

        m.def("create_mvm_descriptor",
              [](int numRows, int numCols, int batchSize, int miniBatchSize, int contextLen, gpu_ops::ElementType f_type)
              {
                  return gpu_ops::PackDescriptor(gpu_ops::MFMDescriptor{
                      numRows, numCols, batchSize, miniBatchSize, contextLen, f_type});
              });

        pybind11::enum_<gpu_ops::ElementType>(m, "ElementType")
            .value("BF16", gpu_ops::ElementType::BF16)
            .value("F32", gpu_ops::ElementType::F32);
    }
} // namespace