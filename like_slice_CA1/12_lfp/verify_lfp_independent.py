"""
Independent re-implementation of extracellular potential (LFP forward model)
from first principles. Written WITHOUT reading the project's own code.

Physics basis (quasi-static, homogeneous isotropic ohmic medium):
  The extracellular potential from a point current source I at distance r is
  the Green's function of Poisson's equation grad.(sigma grad V) = -I delta(r):
      V(r) = I / (4 pi sigma r)

Units bookkeeping (to get mV from nA, um, S/m):
  I [nA] = I * 1e-9 A
  r [um] = r * 1e-6 m
  sigma  [S/m]
  V = I/(4 pi sigma r) = (I*1e-9) / (4 pi sigma * r*1e-6)  [Volt]
    = (1e-9/1e-6) * I/(4 pi sigma r)  = 1e-3 * I/(4 pi sigma r)  [V]
    = I/(4 pi sigma r)  [mV]          (with I in nA, r in um, sigma in S/m)
  So the numeric expression I[nA]/(4 pi sigma r[um]) is DIRECTLY in millivolts.
  (2.6526e-3 mV = 2.6526 uV for the litmus case.)
"""
import numpy as np

FOUR_PI = 4.0 * np.pi


# ---------------------------------------------------------------------------
# 1) Point-Source Approximation (PSA)
# ---------------------------------------------------------------------------
def psa_potential(I_nA, src_xyz, elec_xyz, sigma=0.3):
    """Sum of point-source Green's functions. Returns V in mV.

    I_nA     : array (N,) currents [nA]
    src_xyz  : array (N,3) source midpoints [um]
    elec_xyz : array (3,)  electrode position [um]
    sigma    : conductivity [S/m]
    """
    I_nA = np.atleast_1d(np.asarray(I_nA, float))
    src = np.atleast_2d(np.asarray(src_xyz, float))
    r = np.linalg.norm(src - np.asarray(elec_xyz, float), axis=1)  # [um]
    return np.sum(I_nA / (FOUR_PI * sigma * r))  # mV


# ---------------------------------------------------------------------------
# 2) Line-Source Approximation (LSA)
# ---------------------------------------------------------------------------
# Derivation (my own):
#   A segment is a straight line from a (start) to b (end), length L,
#   carrying total current I distributed uniformly -> line density lambda = I/L.
#   Parametrize the line by arclength s in [0, L] along unit vector u=(b-a)/L.
#   A point on the line: p(s) = a + s*u.
#   Decompose the electrode position relative to the line:
#       w = elec - a
#       s0 = w . u        (projection of electrode onto the line axis; the foot
#                          of the perpendicular is at arclength s0 from a)
#       rho = |w - s0*u|  (perpendicular distance from electrode to the line)
#   Distance from electrode to p(s): |elec - p(s)|^2 = rho^2 + (s - s0)^2.
#   Potential:
#       V = lambda/(4 pi sigma) * INT_0^L ds / sqrt(rho^2 + (s-s0)^2)
#   The integral is a standard arcsinh:
#       INT ds/sqrt(rho^2 + t^2) = arcsinh(t/rho) = ln(t + sqrt(t^2+rho^2))
#   Evaluate t from (0 - s0) to (L - s0):
#       V = I/(4 pi sigma L) * [ arcsinh((L-s0)/rho) - arcsinh((-s0)/rho) ]
#   Equivalent log form (Holt & Koch 1999 style):
#       V = I/(4 pi sigma L) * ln( (sqrt(h^2+rho^2) - h) / (sqrt(l^2+rho^2) - l) )
#       with l = -s0 (signed dist to near end), h = L - s0 (signed dist to far end)
#   I implement BOTH and also a brute-force numerical integral as a cross-check.

def lsa_potential_arcsinh(I_nA, a_xyz, b_xyz, elec_xyz, sigma=0.3):
    """LSA via analytic arcsinh antiderivative. Returns V in mV."""
    a = np.asarray(a_xyz, float)
    b = np.asarray(b_xyz, float)
    elec = np.asarray(elec_xyz, float)
    ab = b - a
    L = np.linalg.norm(ab)
    if L == 0.0:
        # degenerate -> fall back to point source at a
        r = np.linalg.norm(elec - a)
        return I_nA / (FOUR_PI * sigma * r)
    u = ab / L
    w = elec - a
    s0 = np.dot(w, u)                     # foot-of-perpendicular arclength
    perp = w - s0 * u
    rho = np.linalg.norm(perp)            # perpendicular distance
    if rho < 1e-12:
        # electrode lies on the line axis -> arcsinh singular; use log-distance form
        # V = I/(4 pi sigma L) * ln|(L-s0)/(-s0)| (careful with sign / on-segment)
        # handle by tiny offset to avoid singularity
        rho = 1e-9
    term = np.arcsinh((L - s0) / rho) - np.arcsinh((-s0) / rho)
    return I_nA / (FOUR_PI * sigma * L) * term  # mV


def lsa_potential_numeric(I_nA, a_xyz, b_xyz, elec_xyz, sigma=0.3, n=200001):
    """LSA via brute-force numerical integration (trapezoid). Independent check."""
    a = np.asarray(a_xyz, float)
    b = np.asarray(b_xyz, float)
    elec = np.asarray(elec_xyz, float)
    s = np.linspace(0.0, 1.0, n)                 # fractional arclength
    pts = a[None, :] + s[:, None] * (b - a)[None, :]
    d = np.linalg.norm(pts - elec[None, :], axis=1)  # [um]
    L = np.linalg.norm(b - a)
    lam = I_nA / L                                # line density [nA/um]
    # integral over physical arclength ds = L * d(frac)
    integrand = 1.0 / d
    integral = np.trapezoid(integrand, s) * L    # INT ds/dist
    return lam / (FOUR_PI * sigma) * integral     # mV


# ===========================================================================
# VALIDATION CASES
# ===========================================================================
def sep(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

sigma = 0.3

# --- Case (a): single point current, I=1 nA, r=100 um -----------------------
sep("CASE (a): single point source, I=1nA, r=100um, sigma=0.3")
I = 1.0
r = 100.0
theory_mV = I / (FOUR_PI * sigma * r)      # analytic
theory_uV = theory_mV * 1e3
# place source at origin, electrode 100um away on x
psa_mV = psa_potential([I], [[0, 0, 0]], [r, 0, 0], sigma)
print(f"  analytic 1/(4 pi sigma r) = {theory_mV:.10e} mV = {theory_uV:.6f} uV")
print(f"  my PSA implementation     = {psa_mV*1e3:.6f} uV")
print(f"  reported project litmus   = 2.6526 uV")
print(f"  abs diff vs theory        = {abs(psa_mV - theory_mV):.3e} mV")
print(f"  match theory?             = {np.isclose(psa_mV, theory_mV, rtol=1e-12)}")
print(f"  match reported 2.6526uV?  = {np.isclose(psa_uV := psa_mV*1e3, 2.6526, atol=5e-4)}")

# --- Case (b): 20um segment, electrode 500um to the side (far field) ---------
sep("CASE (b): L=20um segment, electrode 500um lateral (FAR field)")
# segment along z from (0,0,-10) to (0,0,+10); midpoint origin; I=1nA
a = np.array([0, 0, -10.0]); b = np.array([0, 0, 10.0])
mid = 0.5 * (a + b)
elec_far = np.array([500.0, 0, 0])
I = 1.0
psa_far = psa_potential([I], [mid], elec_far, sigma)
lsa_far = lsa_potential_arcsinh(I, a, b, elec_far, sigma)
lsa_far_num = lsa_potential_numeric(I, a, b, elec_far, sigma)
rel_far = abs(lsa_far - psa_far) / abs(lsa_far)
print(f"  PSA (point at midpoint) = {psa_far*1e3:.8f} uV")
print(f"  LSA (arcsinh)           = {lsa_far*1e3:.8f} uV")
print(f"  LSA (numeric trapz)     = {lsa_far_num*1e3:.8f} uV")
print(f"  LSA arcsinh vs numeric  = {abs(lsa_far-lsa_far_num)/abs(lsa_far):.3e} (should be ~0)")
print(f"  |LSA-PSA|/|LSA| rel diff = {rel_far:.3e}")
print(f"  reported far rel diff    = 6.7e-5")
print(f"  ~agree at far field?     = {rel_far < 1e-3}")

# --- Case (c): electrode 5um to the side (near field) ------------------------
sep("CASE (c): L=20um segment, electrode 5um lateral (NEAR field)")
elec_near = np.array([5.0, 0, 0])
psa_near = psa_potential([I], [mid], elec_near, sigma)
lsa_near = lsa_potential_arcsinh(I, a, b, elec_near, sigma)
lsa_near_num = lsa_potential_numeric(I, a, b, elec_near, sigma)
rel_near = abs(lsa_near - psa_near) / abs(lsa_near)
print(f"  PSA (point at midpoint) = {psa_near*1e3:.6f} uV")
print(f"  LSA (arcsinh)           = {lsa_near*1e3:.6f} uV")
print(f"  LSA (numeric trapz)     = {lsa_near_num*1e3:.6f} uV")
print(f"  LSA arcsinh vs numeric  = {abs(lsa_near-lsa_near_num)/abs(lsa_near):.3e} (should be ~0)")
print(f"  |LSA-PSA|/|LSA| rel diff = {rel_near*100:.4f} %")
print(f"  reported near rel diff   = 27.8 %")
print(f"  differ at near field?    = {rel_near > 0.05}")

# --- extra: reproduce EXACT reported near-field 27.8% by scanning geometry ---
sep("EXTRA: sensitivity of near-field %% to exact perpendicular distance")
for d_elec in [3.0, 4.0, 5.0, 6.0, 7.0]:
    e = np.array([d_elec, 0, 0])
    p = psa_potential([I], [mid], e, sigma)
    l = lsa_potential_arcsinh(I, a, b, e, sigma)
    print(f"  perp={d_elec:4.1f}um  PSA={p*1e3:8.4f}uV  LSA={l*1e3:8.4f}uV  reldiff={abs(l-p)/abs(l)*100:6.3f}%")
