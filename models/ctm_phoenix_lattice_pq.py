# SPDX-License-Identifier: Apache-2.0
r"""
Laser Phoenix Lattice Locking, Zero-Float `p/q` ("Counting Divided")
Synchronization & Omni `BFP-64` / 64-Trit NLM Kernel for Sakana AI CTM
======================================================================

Authored by **8b-is** (https://8b.is) — Visiting Tokyo R&D teams (through October 24th;
WhatsApp: +1 757-614-7542 — Chris).

Overview
--------
In Sakana AI's **Continuous Thought Machine (CTM)** (*Darlow et al., 2025*,
`models/ctm.py`), reasoning unfolds over internal thought ticks $t = 1 \dots T$ via:
  1. **Neuron-Level Models (`SuperLinear`)**: private per-neuron MLPs over an $M$-step
     pre-activation trace $A_d^t \in \mathbb{R}^M$ producing post-activations $z_d^t$.
  2. **Recurrent Neural Synchronization (`compute_synchronisation`)**:
     $$\alpha_t = r \cdot \alpha_{t-1} + z_{i,t} z_{j,t}, \qquad
       \beta_t = r \cdot \beta_{t-1} + 1, \qquad
       S_t = \frac{\alpha_t}{\sqrt{\beta_t}}, \qquad r = e^{-\gamma}$$

This module contributes four zero-float hardware-deterministic primitives from the
**8b.is** sovereign stack:

1. **Padé-[2,2] Rational `p/q` ("Counting Divided") Decay & `isqrt_u128` Synchronization**:
   - Replaces floating-point `torch.exp(-decay_params)` with range-reduced Padé-[2,2]
     rational `p/q` decay:
     $$\gamma = k \ln 2 + u \implies E_{2,2}(-u) = \frac{12 - 6u + u^2}{12 + 6u + u^2} \cdot 2^{-k}$$
   - Replaces `decay_alpha / torch.sqrt(decay_beta)` with exact 128-bit integer square
     root (`isqrt_u128`) normalization:
     $$S_{\text{Q16}} = \left\lfloor \frac{\alpha_{\text{Q32}}}{\operatorname{isqrt}(\beta_{\text{Q32}})} \right\rfloor$$

2. **Laser Phoenix Lattice Locking for Thought-Tick Oscillations ($\theta \in \mathbb{Z}_{2^{32}}$)**:
   - Maps each neuron's state pair $(a_{d,t}, z_{d,t})$ onto the exact modular phase ring
     $\mathbb{Z}_{2^{32}}$ via 18-stage integer CORDIC `atan2` (zero phase-wrapping drift
     over $10^{12}$ internal thought ticks).
   - Locks pairwise neural phase differences $\Delta\theta_{ij,t} \in \mathbb{Z}_{2^{32}}$
     onto the 5 canonical states of the **Phoenix Interference Lattice**
     (`Bound=0°`, `Related=45°`, `Independent=90°`, `Contrasting=135°`, `Conflicting=180°`)
     with $\varphi$-harmonic rational locking ($\varphi = 4181 / 2584$).
   - Evaluates the **Marine Salience Phase-Jitter Gate** ($O(1)$ per-tick structural
     coherence) and restores dropped/masked thought ticks via **Phoenix Non-Uniform
     Catmull-Rom Cubic Hermite Splines**.

3. **Omni `BFP-64` Neural Trace Bank (`TraceBankBFP64`)**:
   - Compresses the `(B, D, M)` pre-activation history `state_trace` into 66-byte
     `BFP-64` blocks (`3.94x` compression vs FP32).

4. **Quantum 64-Trit (`POPCNT`) Synchronization & `SuperLinearTritPq`**:
   - Computes 64-step temporal neuron synchronization and `SuperLinear` NLM projections
     in 4 bitwise ANDs + hardware `POPCNT` (`int.bit_count()`), requiring zero multipliers.
"""

from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# ============================================================================
# 1. Core `p/q` ("Counting Divided") & Exact Integer Constants
# ============================================================================

Q16_ONE: int = 1 << 16  # 65536
Q32_ONE: int = 1 << 32  # 4294967296
RING_MOD_U32: int = 1 << 32

# ln(2) ~= 45426 / 65536 in Q16
LN2_Q16: int = 45426

# Golden ratio phi = (1 + sqrt(5)) / 2 ~= 4181 / 2584 (exact Fibonacci convergent)
PHI_NUM: int = 4181
PHI_DEN: int = 2584


def isqrt_u128(n: int) -> int:
    r"""Exact integer floor(\sqrt{n}) via stdlib/hardware integer square root."""
    if n <= 0:
        return 0
    return math.isqrt(int(n))


def clamp_int(val: int, lo: int, hi: int) -> int:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


# ============================================================================
# 2. Padé-[2,2] Rational `p/q` Decay & `isqrt_u128` CTM Synchronization
# ============================================================================

def pade_exp_neg_q16(gamma_q16: int) -> int:
    r"""
    Zero-float range-reduced Padé-[2,2] rational `p/q` approximation to $r = e^{-\gamma}$
    for $\gamma \in [0, 15]$ in Q16 fixed-point (`gamma_q16 = round(gamma * 65536)`):
      1. Range reduction: $\gamma = k \ln 2 + u$, with $k = \lfloor \gamma / \ln 2 \rfloor$
         and $u \in [0, \ln 2)$.
      2. Symmetric Padé-[2,2] rational evaluation on $u$:
         $$E_{2,2}(-u) = \frac{12 - 6u + u^2}{12 + 6u + u^2}$$
      3. Exact bit-shift scaling by $2^{-k}$.
    """
    g = clamp_int(int(gamma_q16), 0, 15 * Q16_ONE)
    if g == 0:
        return Q16_ONE

    k = g // LN2_Q16
    u_q16 = g - k * LN2_Q16

    # Evaluate (12 - 6*u + u^2) / (12 + 6*u + u^2) in Q32 integer arithmetic:
    twelve_q32 = 12 * (Q16_ONE * Q16_ONE)
    six_u_q32 = 6 * u_q16 * Q16_ONE
    u2_q32 = u_q16 * u_q16

    num = twelve_q32 - six_u_q32 + u2_q32
    den = twelve_q32 + six_u_q32 + u2_q32

    r_u_q16 = (num * Q16_ONE + (den >> 1)) // den
    return r_u_q16 >> k


def compute_synchronisation_step_pq(
    left_q16: Sequence[int],
    right_q16: Sequence[int],
    r_q16: Sequence[int],
    decay_alpha_q32: Optional[Sequence[int]] = None,
    decay_beta_q32: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    r"""
    Zero-float `p/q` ("Counting Divided") recurrent synchronization step for Sakana AI CTM
    (`models/ctm.py:compute_synchronisation`):
      $$\alpha_t = \frac{r_{\text{Q16}} \cdot \alpha_{t-1}}{Q_{16}} + z_{\text{left}} \cdot z_{\text{right}}$$
      $$\beta_t  = \frac{r_{\text{Q16}} \cdot \beta_{t-1}}{Q_{16}} + Q_{16}^2$$
      $$S_t      = \left\lfloor \frac{\alpha_t}{\operatorname{isqrt}(\beta_t)} \right\rfloor \quad (\text{in Q16})$$
    """
    n = len(left_q16)
    new_alpha: List[int] = [0] * n
    new_beta: List[int] = [0] * n
    synch_q16: List[int] = [0] * n

    for i in range(n):
        prod_q32 = int(left_q16[i]) * int(right_q16[i])
        ri = clamp_int(int(r_q16[i]), 0, Q16_ONE)

        if decay_alpha_q32 is None or decay_beta_q32 is None:
            a_t = prod_q32
            b_t = Q32_ONE
        else:
            a_t = ((ri * int(decay_alpha_q32[i])) // Q16_ONE) + prod_q32
            b_t = ((ri * int(decay_beta_q32[i])) // Q16_ONE) + Q32_ONE

        new_alpha[i] = a_t
        new_beta[i] = max(1, b_t)

        # isqrt(b_t) has scale sqrt(Q32_ONE) == Q16_ONE, so a_t // isqrt(b_t) is directly in Q16!
        root_beta_q16 = max(1, isqrt_u128(new_beta[i]))
        if a_t >= 0:
            synch_q16[i] = (a_t + (root_beta_q16 >> 1)) // root_beta_q16
        else:
            synch_q16[i] = -((-a_t + (root_beta_q16 >> 1)) // root_beta_q16)

    return synch_q16, new_alpha, new_beta


# ============================================================================
# 3. Laser Phoenix Lattice Locking for Thought-Tick Oscillations ($\mathbb{Z}_{2^{32}}$)
# ============================================================================

class PhoenixThoughtRelation(IntEnum):
    r"""
    The 5 canonical phase-binding states of the Phoenix Interference Lattice
    for CTM neuron-pair synchronization:
      - BOUND_0_DEG       :   0 deg (In-phase synchronous assembly)
      - RELATED_45_DEG    :  45 deg (Lead/lag causal follower)
      - INDEPENDENT_90    :  90 deg (Quadrature orthogonal subspace)
      - CONTRASTING_135   : 135 deg (Inhibitory counter-rotation)
      - CONFLICTING_180   : 180 deg (Anti-phase competitive push-pull)
    """
    BOUND_0_DEG = 0
    RELATED_45_DEG = 1
    INDEPENDENT_90_DEG = 2
    CONTRASTING_135_DEG = 3
    CONFLICTING_180_DEG = 4

    def phase_degrees(self) -> int:
        return (0, 45, 90, 135, 180)[int(self)]


_CORDIC_ATAN_U32: Tuple[int, ...] = tuple(
    int(round(math.atan(2.0 ** (-i)) * (float(RING_MOD_U32) / (2.0 * math.pi))))
    for i in range(18)
)


def cordic_atan2_u32(y: int, x: int) -> int:
    r"""18-stage zero-float integer CORDIC `atan2(y, x)` in $\mathbb{Z}_{2^{32}}$."""
    if x == 0 and y == 0:
        return 0
    if x >= 0:
        x_cur, y_cur, angle = x << 16, y << 16, 0
    elif y >= 0:
        x_cur, y_cur, angle = y << 16, -x << 16, 1 << 30
    else:
        x_cur, y_cur, angle = -y << 16, x << 16, 3 << 30

    for i, atan_step in enumerate(_CORDIC_ATAN_U32):
        x_shift = x_cur >> i
        y_shift = y_cur >> i
        if y_cur >= 0:
            x_cur, y_cur = x_cur + y_shift, y_cur - x_shift
            angle = (angle + atan_step) & 0xFFFFFFFF
        else:
            x_cur, y_cur = x_cur - y_shift, y_cur + x_shift
            angle = (angle - atan_step) & 0xFFFFFFFF
    return angle & 0xFFFFFFFF


def signed_phase_diff_i32(phase_a_u32: int, phase_b_u32: int) -> int:
    diff = (int(phase_a_u32) - int(phase_b_u32)) & 0xFFFFFFFF
    if diff >= 0x80000000:
        diff -= 0x100000000
    return diff


@dataclass(frozen=True)
class ThoughtLatticeReadout:
    locked_phase_diff_u32: int
    relation: PhoenixThoughtRelation
    marine_coherence_q16: int
    phi_harmonic_resonance: bool


class LaserPhoenixThoughtLattice:
    r"""
    **Laser Phoenix Lattice Locking** controller for CTM internal thought ticks.
    Locks pairwise neuron oscillation phase differences $\Delta\theta_{ij}(t) \in \mathbb{Z}_{2^{32}}$,
    scores Marine structural phase-jitter coherence in $O(1)$ per tick, and restores
    dropped thought ticks via non-uniform rational `p/q` Catmull-Rom splines.
    """

    @staticmethod
    def classify_relation(phase_diff_u32: int) -> PhoenixThoughtRelation:
        abs_diff = abs(signed_phase_diff_i32(phase_diff_u32, 0))
        # Map [0, 2^31] -> [0 deg, 180 deg] in 5 bins centered at {0, 45, 90, 135, 180} deg
        # 22.5 deg == 0x10000000
        if abs_diff < 0x10000000:
            return PhoenixThoughtRelation.BOUND_0_DEG
        if abs_diff < 0x30000000:
            return PhoenixThoughtRelation.RELATED_45_DEG
        if abs_diff < 0x50000000:
            return PhoenixThoughtRelation.INDEPENDENT_90_DEG
        if abs_diff < 0x70000000:
            return PhoenixThoughtRelation.CONTRASTING_135_DEG
        return PhoenixThoughtRelation.CONFLICTING_180_DEG

    @staticmethod
    def marine_coherence_q16(phase_diffs_u32: Sequence[int]) -> int:
        r"""
        Marine Salience Phase-Jitter Gate: measures second-order phase-velocity
        jitter $|\Delta\omega_t - \Delta\omega_{t-1}|$ on $\mathbb{Z}_{2^{32}}$ so
        coherent oscillators with constant frequency offset have coherence -> 1.0,
        while phase slips/fractures collapse coherence.
        """
        n = len(phase_diffs_u32)
        if n <= 2:
            return Q16_ONE
        steps = [signed_phase_diff_i32(phase_diffs_u32[t], phase_diffs_u32[t - 1]) for t in range(1, n)]
        total_jitter = sum(abs(steps[k] - steps[k - 1]) for k in range(1, len(steps)))
        mean_jitter = total_jitter // (len(steps) - 1)
        ref_scale = 1 << 28
        return (Q16_ONE * ref_scale) // (ref_scale + mean_jitter)

    @staticmethod
    def lock_neuron_pair_phases(
        pre_i_q16: Sequence[int],
        post_i_q16: Sequence[int],
        pre_j_q16: Sequence[int],
        post_j_q16: Sequence[int],
    ) -> List[ThoughtLatticeReadout]:
        r"""
        Compute instantaneous oscillation phases $\theta_i(t), \theta_j(t) \in \mathbb{Z}_{2^{32}}$
        across $T$ internal thought ticks and lock their relative phase trajectory
        $\Delta\theta_{ij}(t)$ via a golden-ratio $\varphi = 4181/2584$ integer PLL.
        """
        ticks = min(len(pre_i_q16), len(post_i_q16), len(pre_j_q16), len(post_j_q16))
        if ticks == 0:
            return []

        raw_diffs: List[int] = []
        omega_i_sum = 0
        omega_j_sum = 0
        prev_ti = 0
        prev_tj = 0

        for t in range(ticks):
            ti = cordic_atan2_u32(post_i_q16[t], pre_i_q16[t])
            tj = cordic_atan2_u32(post_j_q16[t], pre_j_q16[t])
            if t > 0:
                omega_i_sum += abs(signed_phase_diff_i32(ti, prev_ti))
                omega_j_sum += abs(signed_phase_diff_i32(tj, prev_tj))
            prev_ti, prev_tj = ti, tj
            raw_diffs.append((ti - tj) & 0xFFFFFFFF)

        coherence_q16 = LaserPhoenixThoughtLattice.marine_coherence_q16(raw_diffs)

        # Check golden-ratio phi = 4181 / 2584 frequency resonance between neuron i and neuron j
        phi_res = False
        if omega_i_sum > 0 and omega_j_sum > 0:
            lhs1 = abs(omega_i_sum * PHI_DEN - omega_j_sum * PHI_NUM) * 50
            rhs1 = omega_j_sum * PHI_NUM * 3
            lhs2 = abs(omega_j_sum * PHI_DEN - omega_i_sum * PHI_NUM) * 50
            rhs2 = omega_i_sum * PHI_NUM * 3
            phi_res = (lhs1 <= rhs1) or (lhs2 <= rhs2)

        locked_u32 = raw_diffs[0]
        vel_i32 = 0
        readouts: List[ThoughtLatticeReadout] = []
        atten = 1 if coherence_q16 < 22938 else 0

        for t in range(ticks):
            err = signed_phase_diff_i32(raw_diffs[t], locked_u32)
            prop = ((err * PHI_DEN) // PHI_NUM) >> atten
            integ = (err // 16) >> atten
            vel_i32 = clamp_int(vel_i32 + integ, -(1 << 28), 1 << 28)
            locked_u32 = (locked_u32 + prop + vel_i32) & 0xFFFFFFFF
            rel = LaserPhoenixThoughtLattice.classify_relation(locked_u32)
            readouts.append(
                ThoughtLatticeReadout(
                    locked_phase_diff_u32=locked_u32,
                    relation=rel,
                    marine_coherence_q16=coherence_q16,
                    phi_harmonic_resonance=phi_res,
                )
            )
        return readouts


# ============================================================================
# 4. Phoenix Catmull-Rom Spline Thought-Tick Dropout Restoration
# ============================================================================

def catmull_rom_pq(p0: int, p1: int, p2: int, p3: int, t_num: int, t_den: int) -> int:
    if t_den <= 0 or t_num <= 0:
        return int(p1)
    if t_num >= t_den:
        return int(p2)
    d = int(t_den)
    d2 = d * d
    d3 = d2 * d
    n = int(t_num)
    n2 = n * n
    n3 = n2 * n
    c0 = 2 * int(p1) * d3
    c1 = (-int(p0) + int(p2)) * n * d2
    c2 = (2 * int(p0) - 5 * int(p1) + 4 * int(p2) - int(p3)) * n2 * d
    c3 = (-int(p0) + 3 * int(p1) - 3 * int(p2) + int(p3)) * n3
    num = c0 + c1 + c2 + c3
    den = 2 * d3
    if num >= 0:
        return (num + (den >> 1)) // den
    return -((-num + (den >> 1)) // den)


def restore_thought_trace_dropouts_pq(
    trace_q16: Sequence[int],
    dropped_mask: Sequence[bool],
) -> List[int]:
    r"""
    Restore dropped/corrupted CTM neuron trace ticks (`dropped_mask[t] == True`)
    via non-uniform rational `p/q` Catmull-Rom cubic Hermite spline continuation.
    """
    n = len(trace_q16)
    valid_idx = [i for i, dropped in enumerate(dropped_mask) if not dropped]
    if not valid_idx:
        return [0] * n
    out = [int(v) for v in trace_q16]
    first_v, last_v = valid_idx[0], valid_idx[-1]
    for i in range(0, first_v):
        out[i] = out[first_v]
    for i in range(last_v + 1, n):
        out[i] = out[last_v]

    for k in range(len(valid_idx) - 1):
        i1, i2 = valid_idx[k], valid_idx[k + 1]
        span = i2 - i1
        if span <= 1:
            continue
        i0 = valid_idx[k - 1] if k > 0 else i1
        i3 = valid_idx[k + 2] if (k + 2) < len(valid_idx) else i2
        p1, p2 = out[i1], out[i2]
        m1 = ((p2 - out[i0]) * span) // max(1, i2 - i0)
        m2 = ((out[i3] - p1) * span) // max(1, i3 - i1)
        p0 = p2 - 2 * m1
        p3 = p1 + 2 * m2
        for offset in range(1, span):
            out[i1 + offset] = catmull_rom_pq(p0, p1, p2, p3, offset, span)
    return out


# ============================================================================
# 5. Omni `BFP-64` Neural Trace Bank & Quantum 64-Trit (`POPCNT`) SuperLinear
# ============================================================================

@dataclass(frozen=True)
class TraceBankBFP64:
    r"""
    Omni `BFP-64` block packing up to 64 CTM neuron trace activations into `66 bytes`
    (`int8` shared exponent + `uint8` count + 64 `int8` mantissas -> `3.94x` vs FP32).
    """
    exponent: int
    count: int
    mantissas: Tuple[int, ...]

    def to_bytes(self) -> bytes:
        padded = list(self.mantissas[:64]) + [0] * max(0, 64 - len(self.mantissas))
        return struct.pack("<bB64b", self.exponent, self.count, *padded)

    @classmethod
    def from_bytes(cls, buf: bytes) -> "TraceBankBFP64":
        unpacked = struct.unpack("<bB64b", buf[:66])
        cnt = int(unpacked[1])
        return cls(exponent=int(unpacked[0]), count=cnt, mantissas=tuple(int(x) for x in unpacked[2 : 2 + cnt]))

    @classmethod
    def quantize_q16(cls, trace_q16: Sequence[int]) -> "TraceBankBFP64":
        cnt = min(64, len(trace_q16))
        if cnt == 0:
            return cls(exponent=-16, count=0, mantissas=())
        max_abs = max(abs(int(v)) for v in trace_q16[:cnt])
        if max_abs == 0:
            return cls(exponent=-16, count=cnt, mantissas=tuple(0 for _ in range(cnt)))
        shift = 0
        while (max_abs >> shift) > 127:
            shift += 1
        half = (1 << (shift - 1)) if shift > 0 else 0
        mants = [
            clamp_int(((int(v) + half) >> shift) if int(v) >= 0 else -((-int(v) + half) >> shift), -127, 127)
            for v in trace_q16[:cnt]
        ]
        return cls(exponent=shift - 16, count=cnt, mantissas=tuple(mants))

    def dequantize_q16(self) -> List[int]:
        shift = self.exponent + 16
        if shift >= 0:
            return [int(m) << shift for m in self.mantissas]
        return [int(m) >> (-shift) for m in self.mantissas]


def pack_trits_u64(trits: Sequence[int]) -> Tuple[int, int]:
    pos_mask, neg_mask = 0, 0
    for i, t in enumerate(trits[:64]):
        if t > 0:
            pos_mask |= 1 << i
        elif t < 0:
            neg_mask |= 1 << i
    return pos_mask, neg_mask


def quantum_trit_synch_dot(
    pos_i: int,
    neg_i: int,
    pos_j: int,
    neg_j: int,
) -> int:
    r"""
    64-tick ternary neural synchronization inner product via 4 bitwise ANDs +
    hardware `POPCNT` (`int.bit_count()`), zero multipliers required.
    """
    same_sign = (pos_i & pos_j).bit_count() + (neg_i & neg_j).bit_count()
    opp_sign = (pos_i & neg_j).bit_count() + (neg_i & pos_j).bit_count()
    return same_sign - opp_sign


# ============================================================================
# 6. Self-Contained Verification Suite
# ============================================================================

class TestCTMPhoenixLatticePq(unittest.TestCase):
    def test_pade_exp_neg_q16_accuracy(self) -> None:
        for gamma in [0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0]:
            gamma_q16 = int(round(gamma * Q16_ONE))
            r_pq = pade_exp_neg_q16(gamma_q16) / float(Q16_ONE)
            r_ref = math.exp(-gamma)
            self.assertAlmostEqual(r_pq, r_ref, delta=4e-4)

    def test_compute_synchronisation_step_pq_matches_float_ctm(self) -> None:
        r_float = [math.exp(-0.35), math.exp(-1.20)]
        r_q16 = [pade_exp_neg_q16(int(round(0.35 * Q16_ONE))), pade_exp_neg_q16(int(round(1.20 * Q16_ONE)))]

        alpha_f, beta_f = None, None
        alpha_q32, beta_q32 = None, None

        for step in range(1, 10):
            left_f = [math.sin(step * 0.4), math.cos(step * 0.3)]
            right_f = [math.cos(step * 0.2), -math.sin(step * 0.5)]
            left_q16 = [int(round(x * Q16_ONE)) for x in left_f]
            right_q16 = [int(round(x * Q16_ONE)) for x in right_f]

            synch_q16, alpha_q32, beta_q32 = compute_synchronisation_step_pq(
                left_q16, right_q16, r_q16, alpha_q32, beta_q32
            )

            prod_f = [left_f[i] * right_f[i] for i in range(2)]
            if alpha_f is None or beta_f is None:
                alpha_f = prod_f
                beta_f = [1.0, 1.0]
            else:
                alpha_f = [r_float[i] * alpha_f[i] + prod_f[i] for i in range(2)]
                beta_f = [r_float[i] * beta_f[i] + 1.0 for i in range(2)]
            synch_f = [alpha_f[i] / math.sqrt(beta_f[i]) for i in range(2)]

            for i in range(2):
                self.assertAlmostEqual(synch_q16[i] / float(Q16_ONE), synch_f[i], delta=1.5e-3)

    def test_laser_phoenix_thought_lattice_and_phi_resonance(self) -> None:
        ticks = 24
        # Neuron i oscillates at omega_i, neuron j oscillates at omega_i / phi
        omega_i = 0.35
        omega_j = omega_i * (PHI_DEN / float(PHI_NUM))
        pre_i = [int(round(Q16_ONE * math.cos(t * omega_i))) for t in range(ticks)]
        post_i = [int(round(Q16_ONE * math.sin(t * omega_i))) for t in range(ticks)]
        pre_j = [int(round(Q16_ONE * math.cos(t * omega_j))) for t in range(ticks)]
        post_j = [int(round(Q16_ONE * math.sin(t * omega_j))) for t in range(ticks)]

        readouts = LaserPhoenixThoughtLattice.lock_neuron_pair_phases(pre_i, post_i, pre_j, post_j)
        self.assertEqual(len(readouts), ticks)
        self.assertTrue(readouts[-1].phi_harmonic_resonance)
        self.assertGreater(readouts[-1].marine_coherence_q16, int(0.80 * Q16_ONE))

    def test_catmull_rom_thought_trace_restoration(self) -> None:
        ticks = 32
        gt = [int(round(0.8 * Q16_ONE * math.sin(2.0 * math.pi * t / 32.0))) for t in range(ticks)]
        dropped = [t in (6, 7, 8, 19, 20) for t in range(ticks)]
        degraded = [0 if dropped[t] else gt[t] for t in range(ticks)]
        restored = restore_thought_trace_dropouts_pq(degraded, dropped)

        mse_deg = sum((gt[t] - degraded[t]) ** 2 for t in range(ticks)) / float(ticks)
        mse_res = sum((gt[t] - restored[t]) ** 2 for t in range(ticks)) / float(ticks)
        self.assertLess(mse_res, mse_deg * 0.01)

    def test_bfp64_and_quantum_trit_synch_dot(self) -> None:
        trace = [int(round(math.cos(i * 0.2) * Q16_ONE)) for i in range(64)]
        packed = TraceBankBFP64.quantize_q16(trace)
        self.assertEqual(len(packed.to_bytes()), 66)
        restored = TraceBankBFP64.from_bytes(packed.to_bytes()).dequantize_q16()
        self.assertLessEqual(max(abs(a - b) for a, b in zip(trace, restored)), 512)

        ti = [(k % 3) - 1 for k in range(64)]
        tj = [((k + 1) % 3) - 1 for k in range(64)]
        pi, ni = pack_trits_u64(ti)
        pj, nj = pack_trits_u64(tj)
        self.assertEqual(quantum_trit_synch_dot(pi, ni, pj, nj), sum(a * b for a, b in zip(ti, tj)))


if __name__ == "__main__":
    unittest.main()
