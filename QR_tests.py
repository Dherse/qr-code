
from typing import Type

import galois
import numpy as np
import matplotlib.pyplot as plt

from QR_code import QR_code


def test_makeGenerator_check_grs():
    exp: np.ndarray = np.array([0, 229, 121, 135, 48, 211,
                                117, 251, 126, 159, 180, 169,
                                152, 192, 226, 228, 218, 111,
                                0, 117, 232, 87, 96, 227, 21])
    p, m, t = 2, 8, 12
    GF: Type[galois.FieldArray] = galois.GF(p**m)
    a:  galois.Poly = GF.primitive_element
    g:  galois.Poly = QR_code.makeGenerator(p, m, t)

    print(f"QR_Code.makeGenerator(...) check g_rs()\n"
          f"---------------------------------------\n"
          f">> > coeffs g={g.coeffs}\n"
          f"    (expecting coeffs = {GF(a**exp)})\n")


def test_makeGenerator_for_assignment():
    g:  galois.Poly = QR_code.makeGenerator(5, 4, 7)

    print(f"QR_Code.makeGenerator(...) for assignment\n"
          f"-----------------------------------------\n"
          f">>> g(x) = {g}\n"
          f"    (expecting g(x) = ?)\n")


def test_dataStream():
    a = "TEST GROUP 21"
    ds = QR_code.generate_dataStream(a)
    b = QR_code.read_dataStream(ds)

    print(f"QR_Code.generate_dataStream(...)\n"
          f"--------------------------------\n"
          f"bitstream = {ds}\n"
          f">>> b = '{b}'\n"
          f"    (expecting b = '{a}')\n")


def test_data():
    qr = QR_code("Q", "optimal")

    a = "TEST GROUP 21ABCAE AAZR4"
    enc = qr.encodeData(QR_code.generate_dataStream(a))
    dec = qr.decodeData(enc)
    b = QR_code.read_dataStream(dec)

    print(f"QR_Code.generate_dataStream(...)\n"
          f"--------------------------------\n"
          f"encode = {enc}\n"
          f"decode = {dec}\n"
          f">>> b = '{b}'\n"
          f"    (expecting b = '{a}')\n")


def test_decodeRS_from_example_forney():
    p, m, n, t = 2, 4, 15, 3
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    a:          galois.Poly = GF.primitive_element
    r:          galois.Poly = galois.Poly([0, 0, 0, a**7, 0, 0, 0, 0, 0, 0, a**11], field=GF, order="asc")
    g:          galois.Poly = QR_code.makeGenerator(p, m, t)
    k:          int = n-g.degree

    print(f"QR_Code.decodeRS(...) from Forney exmaple (p. 82)\n"
          f"-------------------------------------------------\n"
          f"GF = {GF}\n"
          f"p(x) = {prim_poly}\n"
          f"g(x) = {g}\n"
          f"r(x) = {r}\n"
          f">>> b(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, k, g, 1))}\n"
          f"    (expecting b(x) = 0)\n")


def test_decodeRS_from_example_bma():
    p, m, n, t = 2, 4, 15, 3
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    r:          galois.Poly = galois.Poly([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], field=GF, order="asc")
    g:          galois.Poly = QR_code.makeGenerator(p, m, t)
    k:          int = n - g.degree

    print(f"QR_Code.decodeRS(...) from BMA exmaple (p. 78)\n"
          f"----------------------------------------------\n"
          f"GF = {GF}\n"
          f"p(x) = {prim_poly}\n"
          f"g(x) = {g}\n"
          f"r(x) = {r}\n"
          f">>> b(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, k, g, 1))}\n"
          f"    (expecting b(x) = 0)\n")


def test_decodeRS_from_exercies_11():
    p, m, n, k = 2, 4, 15, 9
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    a:          galois.Poly = GF.primitive_element
    r:          galois.Poly = galois.Poly([a, a**7], field=GF, order="asc")
    g:          galois.Poly = galois.Poly.Roots([a, a**2, a**3, a**4, a**5, a**6], field=GF)

    print(f"QR_Code.decodeRS(...) from exercise 11\n"
          f"--------------------------------------\n"
          f"GF = {GF}\n"
          f"p(x) = {prim_poly}\n"
          f"g(x) = {g}\n"
          f"r(x) = {r}\n"
          f">>> b(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, k, g, 1))}\n"
          f"    (expecting b(x) = 0)\n")


def test_decodeRS_for_assignment():
    p, m, t = 5, 4, 7
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    g:          galois.Poly = QR_code.makeGenerator(p, m, t)
    n:          int = p**m-1
    k:          int = n-g.degree

    print(f"QR_Code.decodeRS(...) for assignment\n"
          f"------------------------------------\n"
          f"GF = {GF}\n"
          f"p(x) = {prim_poly}\n"
          f"g(x) = {g}")
    for i in range(10):
        r: galois.Poly = galois.Poly([j < i for j in range(10)], field=GF, order="asc")
        b: galois.Poly = galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, k, g), field=GF)
        print(f" - r(x) = {r}\n"
              f"   >>> b(x) = {b}\n"
              f"       (expecting b(x) {'!'*(7<i)}= 0)")
    print()


def test_encodeformat():
    # Test case 1 (page 82 of QR_specification)
    level1 = 'M'
    mask_pattern1 = [1, 0, 1]
    expected_output1 = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=int)
    result1 = QR_code.encodeFormat(level1, mask_pattern1)
    assert np.array_equal(result1, expected_output1), f"Test 1 failed: expected {expected_output1}, got {result1}"

    # Test case 2 (own calculation)
    level2 = 'Q'
    mask_pattern2 = [0, 1, 1]
    expected_output2 = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0], dtype=int)
    result2 = QR_code.encodeFormat(level2, mask_pattern2)
    assert np.array_equal(result2, expected_output2), f"Test 2 failed: expected {expected_output2}, got {result2}"


def plot_decodeRS_for_assignment():
    qr: QR_code = QR_code('Q', "optimal")
    b:  str = "GROUP 21 : 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ +*. -/% $"
    # b:  str = "GROUP 21"
    c:  np.ndarray = QR_code.generate_dataStream(b)
    s:  np.ndarray = qr.encodeData(c)

    N: int = 100
    P: list[float] = [5e-3, 1e-2, 2e-2, 5e-2, 2e-1, 1e-1]
    COLORS: list[str] = ["red", "orange", "gold", "lime", "cyan", "blue"]
    scatter = plt.subplot2grid((5, 3), (0, 0), rowspan=3, colspan=3)
    for i, (p, color) in enumerate(zip(P, COLORS)):
        Y: list[float] = []
        L: int = 0
        for _ in range(N):
            try:
                r:        np.ndarray = qr.decodeData((np.random.random(size=s.shape) < p) ^ s)
                length_c: int = len(c)
                length_r: int = len(r)
                c_r:      int = length_c - length_r
                if c_r < 0:
                    t: float = -c_r + np.sum(r[:length_c] ^ c)
                elif 0 < c_r:
                    t: float = c_r + np.sum(r ^ c[:length_r])
                else:
                    t: float = np.sum(r ^ c)
                Y.append(t/len(c))
                L += 1
            except Exception:
                pass

        counts, bins = np.histogram(Y, bins=20)
        hist = plt.subplot2grid((5, 3), (3 + (i // 3), i % 3))
        hist.hist(bins[:-1], bins, weights=counts, color=color)
        hist.set_title(rf"$p = {p}$")
        hist.set_xlim(left=0.0)

        X: list[float] = len(counts)*[i]
        scatter.scatter(X, [0.5 * (bins[j] + bins[j+1]) for j in range(len(counts))], color=color, s=10*counts)

    # scatter.set_xticklabels(P)
    scatter.set_ylim(bottom=0.0)
    plt.tight_layout()
    plt.show()


def test_full_qrcode():
    b:  str = "GROUP 21 : 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ +*. -/% $"
    qr: np.ndarray = QR_code('Q', "optimal").generate(b)
    result: str = QR_code.read(qr)
    print(f"QR code result = {result}\n"
          f"    (expecting {b}")


if __name__ == "__main__":
    # test_makeGenerator_check_grs()
    # test_makeGenerator_for_assignment()
    # test_dataStream()
    # test_data()
    # test_decodeRS_from_example_forney()
    # test_decodeRS_from_example_bma()
    # test_decodeRS_from_exercies_11()
    # test_decodeRS_for_assignment()
    # test_encodeformat()
    # plot_decodeRS_for_assignment()
    test_full_qrcode()
