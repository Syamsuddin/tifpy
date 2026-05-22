"""
demo_step1.py — Demonstrasi & Verifikasi Tahap 1 TIF-SIG
=========================================================

Menjalankan rangkaian uji konsistensi atas konstruksi Hodge Laplacian
dan substrat ℂ⊗ℍ. Setiap uji mencetak status LULUS / GAGAL.

Uji yang dijalankan:
  Bagian A — Hodge Laplacian skalar & teorema Hodge (dim ker L_k = β_k)
    A1. Segitiga berlubang (topologi: lingkaran S¹)
    A2. Segitiga terisi (topologi: cakram)
    A3. Permukaan tetrahedron (topologi: sphere S²)
    A4. Torus 3×3 (topologi: torus T², β₁ = 2)
  Bagian B — Substrat ℂ⊗ℍ & connection Laplacian
    B1. Konsistensi: self-adjoint & positive semi-definite
    B2. Koneksi trivial → dim ker = 2·β₀
    B3. Invariansi gauge: koneksi gauge-trivial → dim ker = 2·β₀
    B4. Holonomi: koneksi acak → kernel mengecil (preview emergensi gauge)
    B5. Pohon (tanpa loop) → koneksi acak tetap dim ker = 2 (β₁ = 0)
"""

import numpy as np
import tifpy as tp

# ----------------------------------------------------------------
# Utilitas pelaporan
# ----------------------------------------------------------------
RESULTS = []

def check(name, condition, detail=""):
    status = "LULUS" if condition else "GAGAL"
    RESULTS.append(condition)
    mark = "[+]" if condition else "[X]"
    line = f"  {mark} {status:6s} | {name}"
    if detail:
        line += f"\n              {detail}"
    print(line)

def header(text):
    print("\n" + "=" * 70)
    print("  " + text)
    print("=" * 70)


# ================================================================
# BAGIAN A — HODGE LAPLACIAN SKALAR & TEOREMA HODGE
# ================================================================
header("BAGIAN A — Hodge Laplacian Skalar & Teorema Hodge")
print("  Memverifikasi: dim ker(L_k) = beta_k untuk setiap dimensi k.")
print("  beta_k dihitung independen via rank operator boundary.")

def test_hodge(name, K, expected_betti):
    print(f"\n  --- {name} ---")
    print(f"      Sel: V={K.n_cells(0)} E={K.n_cells(1)} F={K.n_cells(2)}"
          f"  |  Euler chi = {K.euler_characteristic()}")
    # sanity: boundary-squared-zero
    check(f"{name}: operator boundary memenuhi d.d = 0",
          K.verify_boundary_squared_zero())
    for k in range(3):
        L_k = tp.hodge_laplacian(K, k)
        beta_k = tp.betti_number(K, k)
        dim_ker = tp.kernel_dimension(L_k)
        sa = tp.is_self_adjoint(L_k)
        psd = tp.is_positive_semidefinite(L_k)
        ok = (dim_ker == beta_k) and sa and psd
        check(
            f"{name}: L_{k} self-adjoint+PSD, dim ker = beta_{k}",
            ok,
            f"dim ker(L_{k}) = {dim_ker},  beta_{k} = {beta_k}  "
            f"(harapan: {expected_betti[k]}),  "
            f"self-adjoint={sa}, PSD={psd}"
        )
    # verifikasi Betti sesuai topologi yang diketahui
    actual = tuple(tp.betti_number(K, k) for k in range(3))
    check(f"{name}: bilangan Betti sesuai topologi yang diketahui",
          actual == expected_betti,
          f"Betti terhitung = {actual},  topologi diharapkan = {expected_betti}")


# A1 — Segitiga berlubang (lingkaran S^1): beta = (1,1,0)
K_hollow = tp.SimplicialComplex.from_edges(
    vertices=[0, 1, 2],
    edges=[(0, 1), (1, 2), (0, 2)],
)
test_hodge("A1 Segitiga berlubang (S^1)", K_hollow, (1, 1, 0))

# A2 — Segitiga terisi (cakram): beta = (1,0,0)
K_filled = tp.SimplicialComplex.from_triangles([(0, 1, 2)])
test_hodge("A2 Segitiga terisi (cakram)", K_filled, (1, 0, 0))

# A3 — Permukaan tetrahedron (sphere S^2): beta = (1,0,1)
K_sphere = tp.SimplicialComplex.from_triangles([
    (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
])
test_hodge("A3 Permukaan tetrahedron (S^2)", K_sphere, (1, 0, 1))

# A4 — Torus 3x3 (T^2): beta = (1,2,1)
def torus(m, n):
    def vid(r, c):
        return (r % m) * n + (c % n)
    tris = []
    for r in range(m):
        for c in range(n):
            a, b = vid(r, c), vid(r, c + 1)
            cc, dd = vid(r + 1, c), vid(r + 1, c + 1)
            tris.append((a, b, cc))
            tris.append((b, cc, dd))
    return tp.SimplicialComplex.from_triangles(tris)

K_torus = torus(3, 3)
test_hodge("A4 Torus 3x3 (T^2)", K_torus, (1, 2, 1))


# ================================================================
# BAGIAN B — SUBSTRAT ℂ⊗ℍ & CONNECTION LAPLACIAN
# ================================================================
header("BAGIAN B — Substrat C-tensor-H & Connection Laplacian")
print("  Substrat lokal: modul C^2 atas C-tensor-H = M_2(C).")
print("  Koneksi: elemen SU(2) pada tiap edge. Memverifikasi konsistensi")
print("  formalisme dan hubungan holonomi <-> kernel (preview gauge).")

rng = np.random.default_rng(42)

# Verifikasi representasi kuaternion menghasilkan SU(2)
print("\n  --- B0: Representasi kuaternion C-tensor-H = M_2(C) ---")
all_su2 = all(tp.is_in_su2(tp.random_su2(rng)) for _ in range(20))
check("B0: kuaternion satuan acak -> elemen SU(2) yang valid", all_su2,
      "20 sampel kuaternion satuan; semua menghasilkan matriks unitér det=1")

# Gunakan torus sebagai uji utama (beta_0 = 1, beta_1 = 2 -> holonomi kaya)
K = K_torus
beta0 = tp.betti_number(K, 0)
beta1 = tp.betti_number(K, 1)
print(f"\n  --- Uji pada Torus 3x3 (beta_0={beta0}, beta_1={beta1}) ---")

# B1 + B2 — Koneksi trivial
print("\n  --- B1-B2: Koneksi trivial ---")
conn_triv = tp.trivial_connection(K)
L_triv = tp.connection_laplacian(K, conn_triv)
check("B1: connection Laplacian self-adjoint",
      tp.is_self_adjoint(L_triv))
check("B1: connection Laplacian positive semi-definite",
      tp.is_positive_semidefinite(L_triv),
      f"eigenvalue minimum >= 0 (dalam toleransi)")
dimker_triv = tp.kernel_dimension(L_triv)
check("B2: koneksi trivial -> dim ker = 2 * beta_0",
      dimker_triv == 2 * beta0,
      f"dim ker = {dimker_triv},  2*beta_0 = {2 * beta0}  "
      f"(dua seksi konstan C^2)")

# B3 — Invariansi gauge
print("\n  --- B3: Invariansi gauge ---")
conn_gt = tp.gauge_trivial_connection(K, rng)
L_gt = tp.connection_laplacian(K, conn_gt)
check("B3: connection Laplacian gauge-trivial tetap self-adjoint+PSD",
      tp.is_self_adjoint(L_gt) and tp.is_positive_semidefinite(L_gt))
dimker_gt = tp.kernel_dimension(L_gt)
check("B3: koneksi gauge-trivial -> dim ker = 2*beta_0 (sama dgn trivial)",
      dimker_gt == 2 * beta0,
      f"dim ker = {dimker_gt},  2*beta_0 = {2 * beta0}  "
      f"(holonomi gauge-trivial selalu = I -> spektrum tak berubah)")
# verifikasi holonomi gauge-trivial benar-benar = I pada sebuah loop
loop = [0, 1, 2]  # loop kecil pada torus
H_gt = tp.holonomy(conn_gt, loop)
check("B3: holonomi koneksi gauge-trivial = I (flat)",
      np.allclose(H_gt, np.eye(2), atol=1e-9),
      f"||H - I|| = {np.linalg.norm(H_gt - np.eye(2)):.2e}")

# B4 — Koneksi acak: holonomi non-trivial -> kernel mengecil
print("\n  --- B4: Koneksi acak (holonomi non-trivial) ---")
conn_rand = tp.random_connection(K, rng)
L_rand = tp.connection_laplacian(K, conn_rand)
check("B4: connection Laplacian acak tetap self-adjoint+PSD",
      tp.is_self_adjoint(L_rand) and tp.is_positive_semidefinite(L_rand))
dimker_rand = tp.kernel_dimension(L_rand)
check("B4: koneksi acak -> dim ker < 2*beta_0 (holonomi non-trivial)",
      dimker_rand < 2 * beta0,
      f"dim ker = {dimker_rand} < 2*beta_0 = {2 * beta0}  "
      f"--> holonomi non-trivial mengurangi seksi flat")
# tunjukkan holonomi memang != I
H_rand = tp.holonomy(conn_rand, loop)
check("B4: holonomi koneksi acak != I (non-flat)",
      not np.allclose(H_rand, np.eye(2), atol=1e-6),
      f"||H - I|| = {np.linalg.norm(H_rand - np.eye(2)):.4f}  "
      f"--> ini adalah 'muatan gauge' yang terdeteksi")

# B5 — Pohon: tanpa loop, koneksi acak tetap memberi kernel penuh
print("\n  --- B5: Pohon / path graph (beta_1 = 0, tanpa holonomi) ---")
K_tree = tp.SimplicialComplex.from_edges(
    vertices=list(range(6)),
    edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
)
b0_tree = tp.betti_number(K_tree, 0)
b1_tree = tp.betti_number(K_tree, 1)
conn_tree = tp.random_connection(K_tree, rng)
L_tree = tp.connection_laplacian(K_tree, conn_tree)
dimker_tree = tp.kernel_dimension(L_tree)
check("B5: pohon punya beta_1 = 0 (tak ada loop)",
      b1_tree == 0,
      f"beta_0 = {b0_tree}, beta_1 = {b1_tree}")
check("B5: pada pohon, koneksi acak -> dim ker tetap = 2*beta_0",
      dimker_tree == 2 * b0_tree,
      f"dim ker = {dimker_tree},  2*beta_0 = {2 * b0_tree}  "
      f"--> tanpa loop, tiap koneksi gauge-trivial: holonomi mustahil")


# ================================================================
# RINGKASAN
# ================================================================
header("RINGKASAN VERIFIKASI TAHAP 1")
total = len(RESULTS)
passed = sum(RESULTS)
print(f"\n  Uji LULUS : {passed} / {total}")
if passed == total:
    print("\n  >>> SELURUH UJI KONSISTENSI TAHAP 1 LULUS.")
    print("  >>> Konstruksi Hodge Laplacian pada SIG terbukti konsisten:")
    print("      - Operator boundary memenuhi d.d = 0")
    print("      - Hodge Laplacian self-adjoint dan positive semi-definite")
    print("      - Teorema Hodge terverifikasi: dim ker L_k = beta_k")
    print("      - Substrat C-tensor-H = M_2(C) terimplementasi konsisten")
    print("      - Holonomi SU(2) <-> kernel: preview mekanisme gauge (K4)")
else:
    print(f"\n  >>> {total - passed} UJI GAGAL. Formalisme perlu ditinjau ulang.")
print()

import sys
sys.exit(0 if passed == total else 1)
