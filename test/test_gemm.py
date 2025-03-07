import pytest
from pyblas import sgemm, dgemm, cgemm, zgemm


def test_case1(symm_arr_ones):
    actual = sgemm(order=102,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   b_mat=symm_arr_ones,
                   ldb=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert all(x == 3.0 for x in actual)
    actual = sgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   b_mat=symm_arr_ones,
                   ldb=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert all(x == 3.0 for x in actual)
    actual = sgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   b_mat=symm_arr_ones,
                   ldb=2,
                   beta=5,
                   c_mat=symm_arr_ones,
                   ldc=2)

    assert all(x == 7.0 for x in actual)


def test_case2(sym_arr_ones_lda3, arr_3x2_ones_lda3):
    actual = sgemm(order=102,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=sym_arr_ones_lda3,
                   lda=3,
                   b_mat=sym_arr_ones_lda3,
                   ldb=3,
                   beta=1,
                   c_mat=sym_arr_ones_lda3,
                   ldc=3)

    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 0.0
    assert actual[3] == 3.0
    assert actual[4] == 3.0
    assert actual[5] == 0.0

    actual = sgemm(order=102,
                   transa=111,
                   transb=111,
                   m=3,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=arr_3x2_ones_lda3,
                   lda=3,
                   b_mat=sym_arr_ones_lda3,
                   ldb=3,
                   beta=5,
                   c_mat=sym_arr_ones_lda3,
                   ldc=3)
    # [7.0, 7.0, 0.0, 7.0, 7.0, 0.0, 0.0, 0.0, 0.0]
    assert actual[0] == 7.0
    assert actual[1] == 7.0
    assert actual[2] == 0.0
    assert actual[3] == 7.0
    assert actual[4] == 7.0
    assert actual[5] == 0.0
    assert actual[6] == 0.0
    assert actual[7] == 0.0
    assert actual[8] == 0.0

    actual = sgemm(order=102,
                   transa=111,
                   transb=111,
                   m=3,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                   lda=3,
                   b_mat=sym_arr_ones_lda3,
                   ldb=3,
                   beta=5,
                   c_mat=sym_arr_ones_lda3,
                   ldc=3)
    # [7.0, 7.0, 1.0, 7.0, 7.0, 1.0, 0.0, 0.0, 0.0
    assert actual[0] == 7.0
    assert actual[1] == 7.0
    assert actual[2] == 1.0
    assert actual[3] == 7.0
    assert actual[4] == 7.0
    assert actual[5] == 1.0
    assert actual[6] == 0.0
    assert actual[7] == 0.0
    assert actual[8] == 0.0

    actual = sgemm(order=101,
                   transa=111,
                   transb=111,
                   m=3,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                   lda=3,
                   b_mat=sym_arr_ones_lda3,
                   ldb=3,
                   beta=5,
                   c_mat=[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                   ldc=3)
    #[7.0, 7.0, 0.0, 7.0, 7.0, 0.0, 2.0, 2.0, 1.0]
    assert actual[0] == 7.0
    assert actual[1] == 7.0
    assert actual[2] == 0.0
    assert actual[3] == 7.0
    assert actual[4] == 7.0
    assert actual[5] == 0.0
    assert actual[6] == 2.0
    assert actual[7] == 2.0
    assert actual[8] == 1.0


def test_case3(symm_arr_ones):
    actual = dgemm(order=102,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   b_mat=symm_arr_ones,
                   ldb=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert all(x == 3.0 for x in actual)
    actual = dgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   b_mat=symm_arr_ones,
                   ldb=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert all(x == 3.0 for x in actual)
    actual = dgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   b_mat=symm_arr_ones,
                   ldb=2,
                   beta=5,
                   c_mat=symm_arr_ones,
                   ldc=2)

    assert all(x == 7.0 for x in actual)


def test_case4(symm_arr_ones_c):
    actual = cgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   b_mat=symm_arr_ones_c,
                   ldb=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert all(x.real == 3.0 for x in actual)

    actual = cgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   b_mat=symm_arr_ones_c,
                   ldb=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert all(x == 3.0 for x in actual)
    actual = cgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   b_mat=symm_arr_ones_c,
                   ldb=2,
                   beta=complex(5.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)

    assert all(x == 7.0 for x in actual)


def test_case5(symm_arr_ones_c):
    actual = zgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   b_mat=symm_arr_ones_c,
                   ldb=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert all(x.real == 3.0 for x in actual)

    actual = zgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   b_mat=symm_arr_ones_c,
                   ldb=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert all(x == 3.0 for x in actual)
    actual = zgemm(order=101,
                   transa=111,
                   transb=111,
                   m=2,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   b_mat=symm_arr_ones_c,
                   ldb=2,
                   beta=complex(5.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)

    assert all(x == 7.0 for x in actual)
