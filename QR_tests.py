
from typing import Type

import galois
import numpy as np

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


def test_decodeRS_from_example_forney():
    p, m, n, t = 2, 4, 15, 3
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    a:          galois.Poly = GF.primitive_element
    r:          galois.Poly = galois.Poly([0, 0, 0, a**7, 0, 0, 0, 0, 0, 0, a**11], field=GF, order="asc")
    g:          galois.Poly = QR_code.makeGenerator(p, m, t)

    print(f"QR_Code.decodeRS(...) from Forney exmaple (p. 82)\n"
          f"-------------------------------------------------\n"
          f"GF = {GF}\n"
          f"p(x) = {prim_poly}\n"
          f"g(x) = {g}\n"
          f"r(x) = {r}\n"
          f">>> c(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, None, g, 1))}\n"
          f"    (expecting c(x) = 0)\n")


def test_decodeRS_from_example_bma():
    p, m, n, t = 2, 4, 15, 3
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    r:          galois.Poly = galois.Poly([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], field=GF, order="asc")
    g:          galois.Poly = QR_code.makeGenerator(p, m, t)

    print(f"QR_Code.decodeRS(...) from BMA exmaple (p. 78)\n"
          f"----------------------------------------------\n"
          f"GF = {GF}\n"
          f"p(x) = {prim_poly}\n"
          f"g(x) = {g}\n"
          f"r(x) = {r}\n"
          f">> > c(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, n-2*t-1, g, 1))}\n"
          f"    (expecting c(x) = 0)\n")


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
          f">>> c(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, k, g, 1))}\n"
          f"    (expecting c(x) = 0)\n")


def test_decodeRS_for_assignment():
    p, m, t = 5, 4, 7
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    g:          galois.Poly = QR_code.makeGenerator(p, m, t)

    print(f"QR_Code.decodeRS(...) for assignment\n"
          f"------------------------------------\n"
          f"GF = {GF}\n"
          f"p(x) = {prim_poly}\n"
          f"g(x) = {g}\n")
    for i in range(10):
        r:          galois.Poly = galois.Poly([j < i for j in range(10)], field=GF, order="asc")
        print(f" - r(x) = {r}\n"
              f"   >>> c(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, None, None, g))}\n"
              f"       (expecting c(x) {'!'*(7<i)}= 0)\n")


if __name__ == "__main__":
    test_makeGenerator_check_grs()
    test_makeGenerator_for_assignment()
    test_dataStream()
    test_decodeRS_from_example_forney()
    test_decodeRS_from_example_bma()
    test_decodeRS_from_exercies_11()
    test_decodeRS_for_assignment()
