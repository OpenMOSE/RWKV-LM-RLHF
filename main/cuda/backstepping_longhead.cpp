#include <torch/extension.h>

struct __nv_bfloat16;
using bf = __nv_bfloat16;

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*s0, bf*y, float*s, float*sa, bf*sT);

void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &s0, torch::Tensor &y, c10::optional<torch::Tensor> s, c10::optional<torch::Tensor> sa, torch::Tensor &sT) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)s0.data_ptr(), (bf*)y.data_ptr(), s.has_value() ? (float*)s.value().data_ptr() : NULL, sa.has_value() ? (float*)sa.value().data_ptr() : NULL, (bf*)sT.data_ptr());
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dsT, float*dw, float*dq, float*dk, bf*dv, float*da, float*db, bf*ds0);

void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dsT, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db, torch::Tensor &ds0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dsT.data_ptr(), (float*)dw.data_ptr(), (float*)dq.data_ptr(), (float*)dk.data_ptr(), (bf*)dv.data_ptr(), (float*)da.data_ptr(), (float*)db.data_ptr(), (bf*)ds0.data_ptr());
}

TORCH_LIBRARY(wind_backstepping_longhead, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor s0, Tensor(a!) y, Tensor? s, Tensor? sa, Tensor(d!) sT) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor sa, Tensor dsT, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) da, Tensor(f!) db, Tensor(g!) ds0) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping_longhead, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
