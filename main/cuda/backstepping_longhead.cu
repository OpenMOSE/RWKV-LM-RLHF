// Copyright (c) 2024, Johan Sokrates Wind

#include <cuda_bf16.h>
#include <assert.h>

// using bf = __nv_bfloat16;
// #if defined AMD
// #define to_float(x) (x)
// #define to_bf(x) (x)
// #else
// #define to_float(x) __bfloat162float(x)
// #define to_bf(x) __float2bfloat16_rn(x)
// #endif
using bf = __nv_bfloat16;
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return __float2bfloat16(u); }

typedef bf * __restrict__ F_;

constexpr int K = _K_; // Value dim chunksize

// sum "val" in groups of _C_/K threads with stride K. Expects "share" to be shared memory of size _C_
__device__ inline float sum_reduce(float val, float*share) {
    constexpr int ni = K, nj = _C_/K;
    int i = threadIdx.x%K, j = threadIdx.x/K;
    __syncthreads();
    share[j+i*nj] = val;
    __syncthreads();
    if (j == 0) {
        float sum = 0;
#pragma unroll
        for (int l = 0; l < nj; l++) {
            sum += share[l+i*nj];
        }
        share[i*nj] = sum;
    }
    __syncthreads();
    val = share[i*nj];
    __syncthreads();
    return val;
}

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ s0_, bf* y_, float* s_, float* sa_, bf* sT_) {
    constexpr int C = _C_;
    int bb = blockIdx.z, hh = blockIdx.y, basei = blockIdx.x*K, i = threadIdx.x, rowi_ = basei+i%K, basej = i/K*K;

    float state[K];
    for (int j = 0; j < K; j++) {
        state[j] = to_float(s0_[bb*H*C*C + hh*C*C + rowi_*C + basej+j]);
    }
    __shared__ float q[C], k[C], w[C], a[C], b[C], share[C];

    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0;
#pragma unroll
        for (int j = 0; j < K; j++) {
            sa += state[j] * a[basej+j];
        }
        int vind = bb*T*H*C + t*H*C + hh*C + rowi_;
        sa = sum_reduce(sa, share);
        if (basej == 0 && sa_ != NULL) sa_[vind] = sa;

        float v = to_float(v_[vind]), y = 0;
#pragma unroll
        for (int j = 0; j < K; j++) {
            float& s = state[j];
            int j_ = basej+j;
            s = s * w[j_] + sa * b[j_] + v * k[j_];
            y += s * q[j_];
        }
        y = sum_reduce(y, share);
        if (basej == 0) y_[vind] = to_bf(y);

        if ((t+1)%_CHUNK_LEN_ == 0 && s_ != NULL) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + basej*C + rowi_;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }
    }
    for (int j = 0; j < K; j++) {
        sT_[bb*H*C*C + hh*C*C + rowi_*C + basej+j] = to_bf(state[j]);
    }
}

__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s_, float * __restrict__ sa_, F_ dsT_, float* dw_, float* dq_, float* dk_, bf* dv_, float* da_, float* db_, bf* ds0_) {
    constexpr int C = _C_;
    int bb = blockIdx.z, hh = blockIdx.y, basei = blockIdx.x*K, i = threadIdx.x, rowi = i%K, basej = i/K*K;

    float stateT[K], dstate[K], dstateT[K];
    for (int j = 0; j < K; j++) {
        dstate[j] = to_float(dsT_[(bb*H+hh)*C*C + (basei+rowi)*C + basej+j]);
        dstateT[j] = to_float(dsT_[(bb*H+hh)*C*C + (basei+j)*C + i]);
    }
    __shared__ float w[C], q[C], k[C], a[C], b[C], v[K], dy[K], sa[K], dSb_shared[K], share[C];
    float qi, wi, ki, ai, bi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        if (i < K) {
            int vind = ind + basei;
            v[i] = to_float(v_[vind]);
            dy[i] = to_float(dy_[vind]);
            sa[i] = sa_[vind];
        }
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C + basei;
#pragma unroll
            for (int j = 0; j < K; j++) {
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0, iwi = 1.f/wi, dw = 0, dk = 0, db = 0;
#pragma unroll
        for (int j = 0; j < K; j++) {
            dq += stateT[j]*dy[j];
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstateT[j] += qi * dy[j];
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * v[j];
            db += dstateT[j] * sa[j];
        }
        atomicAdd(dq_+ind, dq);
        atomicAdd(dw_+ind, dw * wi * wi_fac);
        atomicAdd(dk_+ind, dk);
        atomicAdd(db_+ind, db);

        float dv = 0, dSb = 0, dyi = dy[rowi];
#pragma unroll
        for (int j = 0; j < K; j++) {
            dstate[j] += dyi * q[basej+j];
            dv += dstate[j] * k[basej+j];
            dSb += dstate[j] * b[basej+j];
        }
        dv = sum_reduce(dv, share);
        dSb = sum_reduce(dSb, share);
        if (basej == 0) {
            dv_[bb*T*H*C + t*H*C + hh*C + basei+rowi] = to_bf(dv);
            dSb_shared[rowi] = dSb;
        }
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < K; j++) {
            da += stateT[j]*dSb_shared[j];
            dstate[j] = dstate[j] * w[basej+j] + dSb * a[basej+j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
        }
        atomicAdd(da_+ind, da);
    }
    for (int j = 0; j < K; j++) {
        ds0_[(bb*H+hh)*C*C + (basei+rowi)*C + basej+j] = to_bf(dstate[j]);
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*s0, bf*y, float*s, float*sa, bf*sT) {
    static_assert(_C_%K == 0, "_C_ must be divisible by 64");
    forward_kernel<<<dim3(_C_/K,H,B), dim3(_C_)>>>(T,H,w,q,k,v,a,b,s0,y,s,sa,sT);
}
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dsT, float*dw, float*dq, float*dk, bf*dv, float*da, float*db, bf*ds0) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<<<dim3(_C_/K,H,B), dim3(_C_)>>>(T,H,w,q,k,v,a,b,dy,s,sa,dsT,dw,dq,dk,dv,da,db,ds0);
}
