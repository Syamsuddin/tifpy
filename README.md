# tifpy — Spectral Information Graph Toolkit

Prototipe komputasional untuk **Teori Informasi Fundamental (TIF)** berbasis
**Postulat Supremasi Informasi (PSI)**. Implementasi Tahap 1 (Step 1) dari
roadmap matematis TIF-SIG v2.

## Status: Tahap 1 — LULUS (32/32 uji konsistensi)

Tahap 1 memverifikasi bahwa konstruksi Hodge Laplacian pada struktur
Spectral Information Graph (SIG) bersifat konsisten secara matematis, dan
bahwa substrat kompleks-kuaternion ℂ⊗ℍ dapat diimplementasikan secara
konkret.

## Wawasan Kunci

**ℂ⊗ℍ ≅ M₂(ℂ).** Kompleksifikasi aljabar kuaternion isomorfik dengan
aljabar matriks kompleks 2×2. Konsekuensinya, modul lokal atas ℂ⊗ℍ
direpresentasikan sebagai ruang ℂ², dan koneksi SU(2) pada SIG menjadi
matriks 2×2 special-unitary. Ini membuat substrat kuaternion TIF-SIG v2
dapat dikomputasi secara langsung.

## Isi Paket

| Berkas | Keterangan |
|---|---|
| `tifpy.py` | Pustaka inti: SimplicialComplex, Hodge Laplacian, aljabar kuaternion, connection Laplacian, uji konsistensi |
| `demo_step1.py` | Skrip verifikasi: 32 uji konsistensi, mencetak LULUS/GAGAL |
| `tifpy_step1.ipynb` | Versi notebook interaktif (sudah tereksekusi) |
| `README.md` | Berkas ini |

## Cara Menjalankan

```bash
pip install numpy
python3 demo_step1.py        # menjalankan 32 uji verifikasi
jupyter notebook tifpy_step1.ipynb   # versi interaktif
```

## Apa yang Diverifikasi Tahap 1

**Bagian A — Hodge Laplacian skalar & teorema Hodge**

Memverifikasi `dim ker(L_k) = β_k` pada empat kompleks dengan topologi
yang diketahui:

- Segitiga berlubang (lingkaran S¹): β = (1, 1, 0)
- Segitiga terisi (cakram): β = (1, 0, 0)
- Permukaan tetrahedron (sphere S²): β = (1, 0, 1)
- Torus 3×3 (torus T²): β = (1, 2, 1)

β_k dihitung secara independen melalui rank operator boundary, sehingga
kesesuaian dengan `dim ker(L_k)` adalah verifikasi nyata teorema Hodge.

**Bagian B — Substrat ℂ⊗ℍ & connection Laplacian**

- Connection Laplacian selalu self-adjoint dan positive semi-definite.
- Koneksi trivial → `dim ker = 2·β₀` (dua seksi konstan ℂ²).
- Koneksi gauge-trivial → `dim ker = 2·β₀` (invariansi gauge terverifikasi).
- Koneksi acak → `dim ker < 2·β₀` (holonomi non-trivial = muatan gauge).
- Pada pohon (β₁ = 0) → koneksi acak tetap `dim ker = 2·β₀`
  (tanpa loop, holonomi mustahil).

Hasil Bagian B adalah **preview mekanisme emergensi gauge** (Konjektur K4
TIF-SIG v2): holonomi SU(2) di sekitar loop SIG adalah konten gauge, dan
hanya loop (β₁) yang dapat membawanya.

## API Ringkas

```python
import tifpy as tp

# Membangun kompleks
K = tp.SimplicialComplex.from_triangles([(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
K = tp.SimplicialComplex.from_edges([0,1,2], [(0,1),(1,2),(0,2)])

# Hodge Laplacian dan topologi
L0 = tp.hodge_laplacian(K, 0)        # Laplacian graf
beta1 = tp.betti_number(K, 1)        # bilangan Betti
dimker = tp.kernel_dimension(L0)     # dim kernel via eigenvalue

# Substrat kuaternion
U = tp.random_su2(rng)               # elemen SU(2) acak
conn = tp.random_connection(K, rng)  # koneksi pada SIG
Lc = tp.connection_laplacian(K, conn)  # connection Laplacian
H = tp.holonomy(conn, [0,1,2])       # holonomi sepanjang loop

# Uji konsistensi
tp.is_self_adjoint(L)
tp.is_positive_semidefinite(L)
```

## Tahap Berikutnya (Roadmap Langkah 2)

- Konstruksi spectral triple diskret (A_G, H_G, D_G).
- Implementasi aksi spektral S[L_G] = Tr f(L_G/Λ²).
- Connection Laplacian untuk dimensi lebih tinggi (L₁, L₂ ter-twist).
- Persamaan gerak variational dari aksi spektral.

## Lisensi

MIT.
