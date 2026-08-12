# -*- coding: utf-8 -*-
"""전규모 청크 메모리 예산의 두 가지 미지수를 실측한다.

Q1. record 중인 h.Vector 를 np.asarray() 하면 **복사**인가 **뷰**인가?
    복사면 청크마다 기록버퍼와 같은 크기의 메모리가 한 번 더 필요하다(= 예산 2배).
Q2. h.Vector.record(ref, Dt) 는 finitialize 시점에 **미리 할당**되는가, 실행하며 커지는가?
    미리 할당이면 psolve 도중 메모리가 안 늘고, 커지는 방식이면 realloc 순간 2배 피크가 생긴다.

실행: <ca1sim>/python.exe 13_net_fepsp/_memprobe.py
"""
import sys
import numpy as np
from neuron import h

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

h.load_file("stdrun.hoc")
h.dt = 0.025
REC_DT = 0.4
TSTOP = 200.0

s = h.Section(name="s")
s.L = s.diam = 20.0
s.insert("hh")
cv = h.CVode()
cv.use_fast_imem(1)
h.cvode_active(0)

v = h.Vector()
v.record(s(0.5)._ref_i_membrane_, REC_DT)

pc = h.ParallelContext()
pc.set_maxstep(10)
h.finitialize(-65.0)

print("=" * 72)
print("Q2. finitialize 직후 벡터 크기:", v.size(), " (0이면 실행하며 커지는 방식)")

pc.psolve(50.0)
n50 = int(v.size())
pc.psolve(TSTOP)
n200 = int(v.size())
print(f"    psolve 50ms 후 {n50}점 · 200ms 후 {n200}점")
print(f"    → 예상 {int(TSTOP/REC_DT)+1}점. {'실행하며 증가(realloc)' if n50 < n200 else '이상'}")

print()
a = np.asarray(v)
b = np.array(v)
print("Q1. np.asarray(Vector):")
print("    dtype", a.dtype, "· shape", a.shape)
print("    a.base is None ?", a.base is None, "  (False면 **뷰**=복사 안 함)")
print("    a.flags.OWNDATA  ?", a.flags.owndata, "  (False면 **뷰**)")

# 결정적 확인: 원본을 바꾸면 a도 바뀌는가?
old = float(v.x[3])
v.x[3] = -12345.0
print("    원본 v.x[3] 변경 후 a[3] =", a[3], "→", "뷰(복사 아님)" if a[3] == -12345.0 else "복사본")
v.x[3] = old

print()
print("    참고: np.array(Vector).flags.owndata =", b.flags.owndata, "(True면 복사)")
try:
    c = v.as_numpy()
    print("    Vector.as_numpy() 사용 가능 · owndata =", c.flags.owndata)
except Exception as e:
    print("    Vector.as_numpy() 없음:", e)

print()
print("Q1b. np.array([asarray(v) for v in vecs]) 는 반드시 새 배열을 만든다(스택) → 복사 1부 발생")
print("=" * 72)
