# Copyright (c) 2024, Johan Sokrates Wind

import torch as th
import triton
import triton.language as tl

@triton.jit
def IND3(a,b,c,nb,nc):
    return (a*nb+b)*nc+c
@triton.jit
def IND4(a,b,c,d,nb,nc,nd):
    return ((a*nb+b)*nc+c)*nd+d
@triton.jit
def IND5(a,b,c,d,e,nb,nc,nd,ne):
    return (((a*nb+b)*nc+c)*nd+d)*ne+e

@triton.jit
def _prod(a,b): return a*b

# inv(I-A) where A is a strictly lower triangular nxn matrix
@triton.jit
def tri_minv(A, n:tl.constexpr, prec:tl.constexpr):
    i = tl.arange(0,n)
    prod = (i[None,:]==i[:,None]).to(tl.float32)
    for j in range(n-1):
        prod += tl_dot(prec, prod, (A*((i[None,:]==j)*(i[:,None]>i[None,:]))).trans())
    return prod.trans()

@triton.autotune(configs=[triton.Config({'dC': dC}, num_stages=1) for dC in [16,32,64]], key=['T','H','C','dT','prec'])
@triton.jit
def fw_attn_triton(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, wq_,wa_,kwi_,bwi_,fw_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr, dC:tl.constexpr):
    tl.static_assert(C%dC == 0)
    bi = tl.program_id(1)
    hi = tl.program_id(0)
    for i0 in range(0,C,dC):
        i = i0+tl.arange(0,dC)[None,:]
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            state = tl.load(s0_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
            tl.store(s_+IND5(bi,hi,0,i.trans(),j, H,T//dT,C,C), state.to(tl.float32))

    for t0 in range(T//dT):
        dt = tl.arange(0,dT)[:,None]
        t = t0*dT+dt
        tl.debug_barrier()
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sq = tl.load(q_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sk = tl.load(k_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            w = (-sw.exp()).exp()
            fw = tl.reduce(w, 0, _prod, keep_dims=True)
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref

            wq = sq * incl_pref
            wa = sa * non_incl_pref
            kwi = sk * inv_incl_pref
            bwi = sb * inv_incl_pref

            tl.store(wq_+IND4(bi,hi,dt,j, H,dT,C), wq.to(tl.float32))
            tl.store(wa_+IND4(bi,hi,dt,j, H,dT,C), wa.to(tl.float32))
            tl.store(kwi_+IND4(bi,hi,dt,j, H,dT,C), kwi.to(tl.float32))
            tl.store(bwi_+IND4(bi,hi,dt,j, H,dT,C), bwi.to(tl.float32))
            tl.store(fw_+IND3(bi,hi,j, H,C), fw.to(tl.float32))
        tl.debug_barrier()

        ab = tl.zeros((dT,dT), tl.float32)
        ak = tl.zeros((dT,dT), tl.float32)
        qb = tl.zeros((dT,dT), tl.float32)
        qk = tl.zeros((dT,dT), tl.float32)
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            ab += tl_dot(prec, wa, bwi.trans())
            ak += tl_dot(prec, wa, kwi.trans())
            qb += tl_dot(prec, wq, bwi.trans())
            qk += tl_dot(prec, wq, kwi.trans())

        mask1 = (t > t.trans())
        mask2 = (t >= t.trans())
        ab *= mask1
        ak *= mask1
        qb *= mask2
        qk *= mask2

        ab_inv = tri_minv(ab, dT, prec)

        for i0 in range(0,C,dC):
            i = i0+tl.arange(0,dC)[None,:]
            sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

            wa_state = tl.zeros((dT,dC), tl.float32)
            wq_state = tl.zeros((dT,dC), tl.float32)
            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                wa_state += tl_dot(prec, wa, state.trans())
                wq_state += tl_dot(prec, wq, state.trans())

            ab_u = tl_dot(prec, ak, sv) + wa_state
            u = tl_dot(prec, ab_inv, ab_u)
            yy = tl_dot(prec, qk, sv) + tl_dot(prec, qb, u) + wq_state
            tl.store(y_+IND4(bi,t,hi,i, T,H,C), yy.to(tl.bfloat16))

            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                fw = tl.load(fw_+IND3(bi,hi,j, H,C)).to(tl.float32)

                state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)

                if t0+1 < T//dT:
                    tl.store(s_+IND5(bi,hi,t0+1,i.trans(),j, H,T//dT,C,C), state.to(tl.float32))
                else:
                    tl.store(sT_+IND4(bi,hi,i.trans(),j, H,C,C), state.to(tl.bfloat16))


@triton.autotune(configs=[triton.Config({'dC': dC}, num_stages=1) for dC in [16,32,64]], key=['T','H','C','dT','prec'])
@triton.jit
def bw_attn_triton(w_,q_,k_,v_,a_,b_, dy_,s_,dsT_,ds_, dw_,dq_,dk_,dv_,da_,db_,ds0_, wq_,wa_,kwi_,bwi_,fw_,u_,dab_u_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr, dC:tl.constexpr):
    tl.static_assert(C%dC == 0)
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    for i0 in range(0,C,dC):
        i = i0+tl.arange(0,dC)[None,:]
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            dstate = tl.load(dsT_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
            tl.store(ds_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.float32))

    for t0 in range(T//dT-1,-1,-1):
        dt = tl.arange(0,dT)[:,None]
        t = t0*dT+dt
        tl.debug_barrier()
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sq = tl.load(q_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sk = tl.load(k_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            w = (-sw.exp()).exp()
            fw = tl.reduce(w, 0, _prod, keep_dims=True)
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref

            wq = sq * incl_pref
            wa = sa * non_incl_pref
            kwi = sk * inv_incl_pref
            bwi = sb * inv_incl_pref

            tl.store(wq_+IND4(bi,hi,dt,j, H,dT,C), wq.to(tl.float32))
            tl.store(wa_+IND4(bi,hi,dt,j, H,dT,C), wa.to(tl.float32))
            tl.store(kwi_+IND4(bi,hi,dt,j, H,dT,C), kwi.to(tl.float32))
            tl.store(bwi_+IND4(bi,hi,dt,j, H,dT,C), bwi.to(tl.float32))
            tl.store(fw_+IND3(bi,hi,j, H,C), fw.to(tl.float32))
        tl.debug_barrier()

        ab = tl.zeros((dT,dT), tl.float32)
        ak = tl.zeros((dT,dT), tl.float32)
        qb = tl.zeros((dT,dT), tl.float32)
        qk = tl.zeros((dT,dT), tl.float32)
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            ab += tl_dot(prec, wa, bwi.trans())
            ak += tl_dot(prec, wa, kwi.trans())
            qb += tl_dot(prec, wq, bwi.trans())
            qk += tl_dot(prec, wq, kwi.trans())

        mask1 = (t > t.trans())
        mask2 = (t >= t.trans())
        ab *= mask1
        ak *= mask1
        qb *= mask2
        qk *= mask2

        ab_inv = tri_minv(ab, dT, prec)

        dab = tl.zeros((dT,dT), tl.float32)
        dak = tl.zeros((dT,dT), tl.float32)
        dqb = tl.zeros((dT,dT), tl.float32)
        dqk = tl.zeros((dT,dT), tl.float32)

        tl.debug_barrier()
        for i0 in range(0,C,dC):
            i = i0+tl.arange(0,dC)[None,:]
            wa_state = tl.zeros((dT,dC), tl.float32)
            bwi_dw_dstate = tl.zeros((dT,dC), tl.float32)
            kwi_dw_dstate = tl.zeros((dT,dC), tl.float32)
            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                dstate = tl.load(ds_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
                wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                fw = tl.load(fw_+IND3(bi,hi,j, H,C)).to(tl.float32)

                wa_state += tl_dot(prec, wa, state.trans())
                bwi_dw_dstate += tl_dot(prec, bwi*fw, dstate.trans())
                kwi_dw_dstate += tl_dot(prec, kwi*fw, dstate.trans())

            sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

            ab_u = tl_dot(prec, ak, sv) + wa_state
            u = tl_dot(prec, ab_inv, ab_u)
            du = tl_dot(prec, qb.trans(), sdy) + bwi_dw_dstate
            dab_u = tl_dot(prec, ab_inv.trans(), du)

            tl.store(u_+IND4(bi,hi,dt,i, H,dT,C), u.to(tl.float32))
            tl.store(dab_u_+IND4(bi,hi,dt,i, H,dT,C), dab_u.to(tl.float32))

            dv = tl_dot(prec, qk.trans(), sdy) + kwi_dw_dstate + tl_dot(prec, ak.trans(), dab_u)
            tl.store(dv_+IND4(bi,t,hi,i, T,H,C), dv.to(tl.bfloat16))

            dab += tl_dot(prec, dab_u, u.trans()) * mask1
            dak += tl_dot(prec, dab_u, sv.trans()) * mask1
            dqb += tl_dot(prec, sdy, u.trans()) * mask2
            dqk += tl_dot(prec, sdy, sv.trans()) * mask2
        tl.debug_barrier()

        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            dy_state = tl.zeros((dT,dC), tl.float32)
            dab_u_state = tl.zeros((dT,dC), tl.float32)
            fw_u_dstate = tl.zeros((dT,dC), tl.float32)
            fw_v_dstate = tl.zeros((dT,dC), tl.float32)
            state_dstate = tl.zeros((1,dC), tl.float32)

            fw = tl.load(fw_+IND3(bi,hi,j, H,C)).to(tl.float32)
            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            for i0 in range(0,C,dC):
                i = i0+tl.arange(0,dC)[None,:]

                u = tl.load(u_+IND4(bi,hi,dt,i, H,dT,C)).to(tl.float32)
                dab_u = tl.load(dab_u_+IND4(bi,hi,dt,i, H,dT,C)).to(tl.float32)
                sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                tl.debug_barrier()
                dstate = tl.load(ds_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
                tl.debug_barrier()

                dab_u_state += tl_dot(prec, dab_u, state)
                fw_u_dstate += fw * tl_dot(prec, u, dstate)
                fw_v_dstate += fw * tl_dot(prec, sv, dstate)
                dy_state += tl_dot(prec, sdy, state)

                state_dstate += tl.sum(state*dstate, axis=0,keep_dims=True)

                dstate = dstate * fw + tl_dot(prec, sdy.trans(), wq) + tl_dot(prec, dab_u.trans(), wa)
                if t0 > 0:
                    tl.store(ds_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.float32))
                else:
                    tl.store(ds0_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.bfloat16))

            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            w = (-sw.exp()).exp()
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref

            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            da = non_incl_pref * (tl_dot(prec, dab, bwi) + tl_dot(prec, dak, kwi) + dab_u_state)
            tl.store(da_+IND4(bi,t,hi,j, T,H,C), da.to(tl.bfloat16))

            dq = incl_pref * (tl_dot(prec, dqb, bwi) + tl_dot(prec, dqk, kwi) + dy_state)
            tl.store(dq_+IND4(bi,t,hi,j, T,H,C), dq.to(tl.bfloat16))

            db = inv_incl_pref * (tl_dot(prec, dab.trans(), wa) + tl_dot(prec, dqb.trans(), wq) + fw_u_dstate)
            tl.store(db_+IND4(bi,t,hi,j, T,H,C), db.to(tl.bfloat16))

            dk = inv_incl_pref * (tl_dot(prec, dak.trans(), wa) + tl_dot(prec, dqk.trans(), wq) + fw_v_dstate)
            tl.store(dk_+IND4(bi,t,hi,j, T,H,C), dk.to(tl.bfloat16))

            dw0 = fw * state_dstate
            for k in range(t0*dT,t0*dT+dT):
                lmask = (t<k).trans()
                A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
                A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
                A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
                A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
                dw = tl.sum(A, axis=0,keep_dims=True) + dw0

                wk = tl.load(w_+IND4(bi,k,hi,j, T,H,C)).to(tl.float32)
                dw *= -wk.exp()
                tl.store(dw_+IND4(bi,k,hi,j, T,H,C), dw.to(tl.bfloat16))


@triton.jit
def tl_dot(prec:tl.constexpr, a, b):
    if prec == 'fp32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=False)
    #elif prec == 'tf32': # This sometimes runs into a bug in the triton language
        #return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=True)
    elif prec == 'bf16':
        return tl.dot(a.to(tl.bfloat16),b.trans().to(tl.bfloat16).trans(), allow_tf32=True)
    else:
        tl.static_assert(False)


class RWKV7_bighead(th.autograd.Function):
    @staticmethod
    def forward(ctx, q,w,k,v,a,b,s0, dot_prec):
        K = 16
        B,T,H,C = w.shape
        assert T%K == 0
        assert C%16 == 0

        assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b,s0])
        assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
        assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
        assert list(s0.shape) == [B,H,C,C]

        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.zeros(B,H,T//K,C,C, dtype=th.float32,device=w.device)
        wq,wa,kwi,bwi = [th.empty(B,H,K,C, dtype=th.float32,device=w.device) for i in range(4)]
        fw = th.empty(B,H,C, dtype=th.float32,device=w.device)
        fw_attn_triton[(H,B)](w,q,k,v,a,b, s0,y,s,sT, wq,wa,kwi,bwi,fw, B,T,H,C,K, dot_prec)
        ctx.dot_prec = dot_prec
        ctx.save_for_backward(w,q,k,v,a,b,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        K = 16
        w,q,k,v,a,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        fw = th.empty(B,H,C, dtype=th.float32,device=w.device)
        ds = th.empty(B,H,C,C, dtype=th.float32,device=w.device)
        wq,wa,kwi,bwi,u,dab_u = [th.empty(B,H,K,C, dtype=th.float32,device=w.device) for i in range(6)]
        bw_attn_triton[(H,B)](w,q,k,v,a,b, dy,s,dsT,ds, dw,dq,dk,dv,da,db,ds0, wq,wa,kwi,bwi,fw,u,dab_u, B,T,H,C,K, ctx.dot_prec)
        return dq,dw,dk,dv,da,db,ds0,None

def attn_triton_bighead(r,w,k,v,a,b, s0 = None, dot_prec='fp32'):
    B,T,H,C = w.shape
    if s0 is None: s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return RWKV7_bighead.apply(r,w,k,v,a,b,s0,dot_prec)

def attn_triton_bighead_bf16(*args): return attn_triton_bighead(*args,dot_prec='bf16')
#def attn_triton_bighead_tf32(*args): return attn_triton_bighead(*args,dot_prec='tf32')
def attn_triton_bighead_fp32(*args): return attn_triton_bighead(*args,dot_prec='fp32')

def attn_triton_bighead_wrap(r,w,k,v,a,b, s0 = None, return_state = False, head_size = 64, dot_prec = 'fp32'):
    B,T,HC = w.shape
    C = head_size
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return RWKV7_bighead.apply(r,w,k,v,a,b,s0,dot_prec)[0].view(B,T,HC)

def RUN_CUDA_RWKV7g(r,w,k,v,a,b,HEAD_SIZE=64, mask=None,  dot_prec = 'fp32'):
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    
    # マスクの前処理をPythonレベルで行う
    if mask is not None:
        # マスクがある場合は以下の項目を直接マスクする
        # 1. QとKの値が0のところを検知
        mask_view = mask.view(B, T, 1, 1)
        
        # 入力テンソルを直接マスクする（注意：元のテンソルを変更する）
        r = r * mask_view
        k = k * mask_view
        v = v * mask_view
        a = a * mask_view
        b = b * mask_view
    
    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return RWKV7_bighead.apply(r,w,k,v,a,b,s0,dot_prec)[0].view(B,T,HC)



def RUN_CUDA_RWKV7g_chunk(r,w,k,v,a,b, HEAD_SIZE, state=None,  dot_prec = 'fp32'):
    #mask and dot_prec is dummy
    B,T,HC = w.shape
    C = HEAD_SIZE
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    if state is None:
        s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    else:
        s0 = state
    y, sT = RWKV7_bighead.apply(r,w,k,v,a,b,s0,dot_prec)
    return y.view(B,T,HC), sT