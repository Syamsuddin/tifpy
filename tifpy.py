"""
tifpy — Spectral Information Graph (SIG) toolkit
================================================
Prototype, Tahap 1 (Step 1) dari roadmap TIF-SIG v2.

Tujuan Tahap 1: memverifikasi bahwa konstruksi Hodge Laplacian pada
struktur SIG bersifat konsisten secara matematis, dan bahwa substrat
kompleks-kuaternion ℂ⊗ℍ dapat diimplementasikan secara konkret.

Wawasan kunci: ℂ⊗ℍ ≅ M₂(ℂ). Kompleksifikasi aljabar kuaternion
isomorfik dengan aljabar matriks kompleks 2×2. Maka modul lokal atas
ℂ⊗ℍ direpresentasikan sebagai ruang ℂ², dan koneksi SU(2) menjadi
matriks 2×2 special-unitary.

Modul ini mengimplementasikan:
  (A) SimplicialComplex dengan operator boundary simplisial
  (B) Hodge Laplacian skalar L_k dan bilangan Betti β_k
  (C) Aljabar kuaternion: kuaternion ↔ M₂(ℂ), SU(2), holonomi
  (D) Connection Laplacian (Hodge Laplacian ter-twist oleh koneksi SU(2))
  (E) Uji konsistensi: self-adjoint, positive semi-definite, teorema Hodge

Penulis: Syamsuddin (kerangka), prototipe implementasi tifpy.
Lisensi: MIT.
"""

import itertools
import numpy as np

# ============================================================
# (A) STRUKTUR: SIMPLICIAL COMPLEX DAN OPERATOR BOUNDARY
# ============================================================

class SimplicialComplex:
    """
    Kompleks simplisial sebagai model konkret SIG.

    Simpleks disimpan sebagai tuple terurut (sorted). Operator boundary
    menggunakan rumus simplisial standar dengan tanda berganti, yang
    MENJAMIN ∂_{k-1} ∘ ∂_k = 0 secara eksak — tanpa perlu mengelola
    orientasi secara manual.

    Parameter
    ---------
    simplices_by_dim : dict {k: list of tuples}
        Daftar simpleks per dimensi. Simpleks-k adalah tuple berisi
        (k+1) indeks simpul. 0-simpleks adalah 1-tuple (v,).
    """

    def __init__(self, simplices_by_dim):
        self.simplices = {}
        for k, simps in simplices_by_dim.items():
            normed = sorted(set(tuple(sorted(s)) for s in simps))
            self.simplices[int(k)] = normed
        self.max_dim = max(self.simplices.keys()) if self.simplices else 0
        for k in range(self.max_dim + 1):
            self.simplices.setdefault(k, [])
        # indeks pencarian cepat: simpleks -> posisi kolom/baris
        self._index = {
            k: {s: i for i, s in enumerate(self.simplices[k])}
            for k in self.simplices
        }

    @classmethod
    def from_triangles(cls, triangles):
        """Bangun kompleks dari daftar segitiga; edge & simpul diturunkan otomatis."""
        tris = set(tuple(sorted(t)) for t in triangles)
        edges, verts = set(), set()
        for t in tris:
            for v in t:
                verts.add(v)
            for e in itertools.combinations(t, 2):
                edges.add(tuple(sorted(e)))
        return cls({
            0: [(v,) for v in sorted(verts)],
            1: sorted(edges),
            2: sorted(tris),
        })

    @classmethod
    def from_edges(cls, vertices, edges):
        """Bangun kompleks-1 (graf) tanpa segitiga — untuk uji topologi murni."""
        return cls({
            0: [(v,) for v in sorted(vertices)],
            1: [tuple(sorted(e)) for e in edges],
            2: [],
        })

    def n_cells(self, k):
        """Jumlah sel berdimensi-k."""
        return len(self.simplices.get(k, []))

    def boundary_matrix(self, k):
        """
        Matriks boundary ∂_k : C_k → C_{k-1}, berbentuk (n_{k-1}, n_k).

        Entri kolom-σ baris-τ adalah (-1)^i bila τ = σ dengan simpul
        ke-i dihapus; 0 selainnya.
        """
        n_k = self.n_cells(k)
        if k <= 0:
            # ∂_0 : C_0 → C_{-1} = 0
            return np.zeros((0, n_k))
        if k not in self.simplices or n_k == 0:
            # tidak ada sel berdimensi-k: ∂_k adalah peta nol
            return np.zeros((self.n_cells(k - 1), n_k))
        n_km1 = self.n_cells(k - 1)
        D = np.zeros((n_km1, n_k))
        for j, sigma in enumerate(self.simplices[k]):
            for i_rm in range(len(sigma)):
                face = sigma[:i_rm] + sigma[i_rm + 1:]
                row = self._index[k - 1].get(face)
                if row is None:
                    raise ValueError(
                        f"Face {face} dari simpleks {sigma} tidak ada di kompleks. "
                        f"Kompleks tidak valid."
                    )
                D[row, j] = (-1) ** i_rm
        return D

    def euler_characteristic(self):
        """Karakteristik Euler χ = Σ (-1)^k n_k."""
        return sum((-1) ** k * self.n_cells(k) for k in range(self.max_dim + 1))

    def verify_boundary_squared_zero(self, tol=1e-10):
        """Verifikasi ∂_{k-1} ∘ ∂_k = 0 untuk semua k (sanity check kompleks)."""
        ok = True
        for k in range(2, self.max_dim + 1):
            D_k = self.boundary_matrix(k)
            D_km1 = self.boundary_matrix(k - 1)
            if D_k.size and D_km1.size:
                prod = D_km1 @ D_k
                if not np.allclose(prod, 0, atol=tol):
                    ok = False
        return ok


# ============================================================
# (B) HODGE LAPLACIAN SKALAR DAN BILANGAN BETTI
# ============================================================

def hodge_laplacian(K, k):
    """
    Hodge Laplacian dimensi-k:  L_k = ∂_k^T ∂_k + ∂_{k+1} ∂_{k+1}^T.

    Bekerja pada ruang C_k. L_0 mereduksi ke Laplacian graf standar.
    """
    D_k = K.boundary_matrix(k)        # C_k → C_{k-1}
    D_kp1 = K.boundary_matrix(k + 1)  # C_{k+1} → C_k
    n_k = K.n_cells(k)
    down = D_k.T @ D_k if D_k.size else np.zeros((n_k, n_k))
    up = D_kp1 @ D_kp1.T if D_kp1.size else np.zeros((n_k, n_k))
    return down + up


def betti_number(K, k, tol=1e-9):
    """
    Bilangan Betti β_k dihitung dari rank operator boundary:
        β_k = (n_k - rank ∂_k) - rank ∂_{k+1}.
    Ini adalah perhitungan INDEPENDEN dari Hodge Laplacian, dipakai
    untuk memverifikasi teorema Hodge: dim ker L_k = β_k.
    """
    D_k = K.boundary_matrix(k)
    D_kp1 = K.boundary_matrix(k + 1)
    n_k = K.n_cells(k)
    rank_k = int(np.linalg.matrix_rank(D_k, tol=tol)) if D_k.size else 0
    rank_kp1 = int(np.linalg.matrix_rank(D_kp1, tol=tol)) if D_kp1.size else 0
    return (n_k - rank_k) - rank_kp1


# ============================================================
# (C) ALJABAR KUATERNION: ℂ⊗ℍ ≅ M₂(ℂ)
# ============================================================

def quaternion_to_matrix(q):
    """
    Representasikan kuaternion q = a + b·i + c·j + d·k sebagai matriks
    kompleks 2×2 (representasi standar ℍ ↪ M₂(ℂ)).

    Kuaternion satuan ↦ elemen SU(2).
    """
    a, b, c, d = q
    return np.array([
        [a + 1j * b,  c + 1j * d],
        [-c + 1j * d, a - 1j * b],
    ], dtype=complex)


def random_su2(rng):
    """Elemen SU(2) acak: kuaternion satuan acak ↦ matriks."""
    v = rng.standard_normal(4)
    v = v / np.linalg.norm(v)
    return quaternion_to_matrix(v)


def is_in_su2(U, tol=1e-9):
    """Periksa apakah U ∈ SU(2): unitér dan determinan = 1."""
    unit = np.allclose(U.conj().T @ U, np.eye(2), atol=tol)
    det1 = abs(np.linalg.det(U) - 1.0) < tol
    return unit and det1


# ============================================================
# (D) CONNECTION LAPLACIAN — HODGE LAPLACIAN TER-TWIST
# ============================================================
# Substrat: tiap simpul membawa modul lokal ℂ² (modul atas ℂ⊗ℍ ≅ M₂(ℂ)).
# Koneksi: tiap edge {i,j} (i<j) membawa U_{ij} ∈ SU(2), transport j→i.
# Connection Laplacian pada simpul:
#     (L^∇ f)(i) = Σ_{j~i} ( f(i) - U_{ij} f(j) )
# berbentuk matriks blok (2n × 2n).

def trivial_connection(K):
    """Koneksi trivial: U_{ij} = I₂ pada setiap edge."""
    return {e: np.eye(2, dtype=complex) for e in K.simplices[1]}


def gauge_trivial_connection(K, rng):
    """
    Koneksi gauge-trivial: U_{ij} = g_i^† g_j untuk g_v ∈ SU(2) acak.
    Holonomi koneksi seperti ini SELALU trivial (≡ I) di setiap loop.
    Dipakai untuk menguji invariansi gauge.
    """
    g = {(v,)[0]: random_su2(rng) for (v,) in K.simplices[0]}
    conn = {}
    for (i, j) in K.simplices[1]:
        conn[(i, j)] = g[i].conj().T @ g[j]
    return conn


def random_connection(K, rng):
    """Koneksi acak: U_{ij} ∈ SU(2) acak independen di tiap edge."""
    return {e: random_su2(rng) for e in K.simplices[1]}


def connection_laplacian(K, connection, d=2):
    """
    Connection Laplacian (twisted Hodge Laplacian) pada 0-sel.

    Parameter
    ---------
    connection : dict {(i,j): U}  untuk i<j, U matriks d×d unitér.
    d : dimensi modul lokal (d=2 untuk substrat ℂ⊗ℍ).

    Mengembalikan matriks Hermitian (n·d × n·d).
    """
    n = K.n_cells(0)
    L = np.zeros((n * d, n * d), dtype=complex)
    deg = np.zeros(n)
    for (i, j) in K.simplices[1]:
        U = connection[(i, j)]
        deg[i] += 1
        deg[j] += 1
        bi, bj = i * d, j * d
        L[bi:bi + d, bj:bj + d] += -U
        L[bj:bj + d, bi:bi + d] += -U.conj().T
    for i in range(n):
        L[i * d:i * d + d, i * d:i * d + d] += deg[i] * np.eye(d)
    return L


def _transport(connection, a, b):
    """Peta transport fiber(a) → fiber(b) sepanjang edge {a,b}."""
    if a < b:
        # U_{ab}: fiber(b)→fiber(a), maka dagger memberi fiber(a)→fiber(b)
        return connection[(a, b)].conj().T
    else:
        # U_{ba}: fiber(a)→fiber(b)
        return connection[(b, a)]


def holonomy(connection, loop):
    """
    Holonomi koneksi sepanjang loop simpul [v0, v1, ..., vk] (kembali ke v0).

    Mengembalikan matriks 2×2: hasil komposisi transport mengelilingi loop.
    Holonomi = I  ⇔  koneksi flat pada loop tersebut.
    """
    H = np.eye(2, dtype=complex)
    full = list(loop) + [loop[0]]
    for a, b in zip(full[:-1], full[1:]):
        H = _transport(connection, a, b) @ H
    return H


# ============================================================
# (E) UJI KONSISTENSI
# ============================================================

def is_self_adjoint(L, tol=1e-9):
    """Periksa L = L^† (Hermitian)."""
    if L.size == 0:
        return True
    return bool(np.allclose(L, L.conj().T, atol=tol))


def is_positive_semidefinite(L, tol=1e-8):
    """Periksa semua eigenvalue ≥ 0 (dalam toleransi)."""
    if L.size == 0:
        return True
    eig = np.linalg.eigvalsh((L + L.conj().T) / 2)
    return bool(eig.min() > -tol)


def kernel_dimension(L, tol=1e-8):
    """Dimensi kernel = jumlah eigenvalue (mendekati) nol."""
    if L.size == 0:
        return 0
    eig = np.linalg.eigvalsh((L + L.conj().T) / 2)
    return int(np.sum(np.abs(eig) < tol))


def spectral_gap(L, tol=1e-8):
    """Spectral gap = eigenvalue positif terkecil (0 bila tak ada)."""
    if L.size == 0:
        return 0.0
    eig = np.sort(np.linalg.eigvalsh((L + L.conj().T) / 2))
    positive = eig[eig > tol]
    return float(positive[0]) if positive.size else 0.0
