
from typing import Type

import galois

from QR_code import QR_code


def test_dataStream():
    a = "TEST GROUP 21"
    ds = QR_code.generate_dataStream(a)
    b = QR_code.read_dataStream(ds)

    print("QR_Code.generate_dataStream(...)")
    print("--------------------------------")
    print(f"bitstream = {ds}")
    print(f">>> b = '{b}'\n    (expecting b = '{a}')")
    print()

def test_decodeRS_from_example_bma():
    p, m, n, t = 2, 4, 15, 3
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    r:          galois.Poly = galois.Poly([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], field=GF, order="asc")
    g:          galois.Poly = QR_code.makeGenerator(p, m, t)

    print("QR_Code.decodeRS(...) from BMA exmaple (p. 78)")
    print("----------------------------------------------")
    print(f"GF = {GF}")
    print(f"p(x) = {prim_poly}")
    print(f"g(x) = {g}")
    print(f"r(x) = {r}")
    print(f">>> c(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, n-2*t-1, g, 1))}\n    (expecting c(x) = 0)")
    print()


def test_decodeRS_from_exercies_11():
    p, m, n, k = 2, 4, 15, 9
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    a:          galois.Poly = GF.primitive_element
    r:          galois.Poly = galois.Poly([a, a**7], field=GF, order="asc")
    g:          galois.Poly = galois.Poly.Roots([a, a**2, a**3, a**4, a**5, a**6], field=GF)

    print("QR_Code.decodeRS(...) from exercise 11")
    print("--------------------------------------")
    print(f"GF = {GF}")
    print(f"p(x) = {prim_poly}")
    print(f"g(x) = {g}")
    print(f"r(x) = {r}")
    print(f">>> c(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, n, k, g, 1))}\n    (expecting c(x) = 0)")
    print()


def test_decodeRS_for_assignment():
    p, m, t = 5, 4, 7
    prim_poly:  galois.Poly = galois.primitive_poly(p, m)
    GF:         Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
    g:          galois.Poly = QR_code.makeGenerator(p, m, t)

    print("QR_Code.decodeRS(...) for assignment")
    print("------------------------------------")
    print(f"GF = {GF}")
    print(f"p(x) = {prim_poly}")
    print(f"g(x) = {g}")
    for i in range(10):
        r:          galois.Poly = galois.Poly([j < i for j in range(10)], field=GF, order="asc")
        print(f" - r(x) = {r}")
        print(f"   >>> c(x) = {galois.Poly(QR_code.decodeRS(r.coeffs, p, m, None, None, g, 1))}\n       (expecting c(x) {'!'*(7<i)}= 0)")


if __name__ == "__main__":
    test_dataStream()
    test_decodeRS_from_example_bma()
    test_decodeRS_from_exercies_11()
    test_decodeRS_for_assignment()
