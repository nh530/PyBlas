import pytest
from pyblas import cher2k, zher2k


def test_case1(symm_arr_ones_c):
    actual = cher2k(order=102,
                    uplo=121,
                    trans=111,
                    n=2,
                    k=2,
                    alpha=complex(1.0, 0.0),
                    a_mat=symm_arr_ones_c,
                    lda=2,
                    b_mat=symm_arr_ones_c,
                    ldb=2,
                    beta=1.0,
                    c_mat=symm_arr_ones_c,
                    ldc=2)
    assert actual[0] == 5.0
    assert actual[1] == 1.0
    assert actual[2] == 5.0
    assert actual[3] == 5.0

    actual = cher2k(order=102,
                    uplo=122,
                    trans=111,
                    n=2,
                    k=2,
                    alpha=complex(1.0, 0.0),
                    a_mat=symm_arr_ones_c,
                    lda=2,
                    b_mat=symm_arr_ones_c,
                    ldb=2,
                    beta=1.0,
                    c_mat=symm_arr_ones_c,
                    ldc=2)
    assert actual[0] == 5.0
    assert actual[1] == 5.0
    assert actual[2] == 1.0
    assert actual[3] == 5.0

    actual = cher2k(order=101,
                    uplo=121,
                    trans=111,
                    n=2,
                    k=2,
                    alpha=complex(1.0, 0.0),
                    a_mat=symm_arr_ones_c,
                    lda=2,
                    b_mat=symm_arr_ones_c,
                    ldb=2,
                    beta=5.0,
                    c_mat=symm_arr_ones_c,
                    ldc=2)
    assert actual[0] == 9.0
    assert actual[1] == 9.0
    assert actual[2] == 1.0
    assert actual[3] == 9.0


def test_case2(symm_arr_ones_c):
    actual = zher2k(order=102,
                    uplo=121,
                    trans=111,
                    n=2,
                    k=2,
                    alpha=complex(1.0, 0.0),
                    a_mat=symm_arr_ones_c,
                    lda=2,
                    b_mat=symm_arr_ones_c,
                    ldb=2,
                    beta=1.0,
                    c_mat=symm_arr_ones_c,
                    ldc=2)
    assert actual[0] == 5.0
    assert actual[1] == 1.0
    assert actual[2] == 5.0
    assert actual[3] == 5.0

    actual = zher2k(order=102,
                    uplo=122,
                    trans=111,
                    n=2,
                    k=2,
                    alpha=complex(1.0, 0.0),
                    a_mat=symm_arr_ones_c,
                    lda=2,
                    b_mat=symm_arr_ones_c,
                    ldb=2,
                    beta=1.0,
                    c_mat=symm_arr_ones_c,
                    ldc=2)
    assert actual[0] == 5.0
    assert actual[1] == 5.0
    assert actual[2] == 1.0
    assert actual[3] == 5.0

    actual = zher2k(order=101,
                    uplo=121,
                    trans=111,
                    n=2,
                    k=2,
                    alpha=complex(1.0, 0.0),
                    a_mat=symm_arr_ones_c,
                    lda=2,
                    b_mat=symm_arr_ones_c,
                    ldb=2,
                    beta=5.0,
                    c_mat=symm_arr_ones_c,
                    ldc=2)
    assert actual[0] == 9.0
    assert actual[1] == 9.0
    assert actual[2] == 1.0
    assert actual[3] == 9.0

