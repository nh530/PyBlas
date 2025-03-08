import pytest
from pyblas import ssyr2k, dsyr2k, csyr2k, zsyr2k


def test_case1(symm_arr_ones):
    actual = ssyr2k(order=102,
                    uplo=121,
                    trans=111,
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
    assert actual[0] == 5.0
    assert actual[1] == 1.0
    assert actual[2] == 5.0
    assert actual[3] == 5.0

    actual = ssyr2k(order=102,
                    uplo=122,
                    trans=111,
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
    assert actual[0] == 5.0
    assert actual[1] == 5.0
    assert actual[2] == 1.0
    assert actual[3] == 5.0

    actual = ssyr2k(order=101,
                    uplo=121,
                    trans=111,
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
    assert actual[0] == 9.0
    assert actual[1] == 9.0
    assert actual[2] == 1.0
    assert actual[3] == 9.0


def test_case2(symm_arr_ones):
    actual = dsyr2k(order=102,
                    uplo=121,
                    trans=111,
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
    assert actual[0] == 5.0
    assert actual[1] == 1.0
    assert actual[2] == 5.0
    assert actual[3] == 5.0

    actual = dsyr2k(order=102,
                    uplo=122,
                    trans=111,
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
    assert actual[0] == 5.0
    assert actual[1] == 5.0
    assert actual[2] == 1.0
    assert actual[3] == 5.0

    actual = dsyr2k(order=101,
                    uplo=121,
                    trans=111,
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
    assert actual[0] == 9.0
    assert actual[1] == 9.0
    assert actual[2] == 1.0
    assert actual[3] == 9.0


def test_case3(symm_arr_ones_c):
    actual = csyr2k(order=102,
                    uplo=121,
                    trans=111,
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
    assert actual[0] == 5.0
    assert actual[1] == 1.0
    assert actual[2] == 5.0
    assert actual[3] == 5.0

    actual = csyr2k(order=102,
                    uplo=122,
                    trans=111,
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
    assert actual[0] == 5.0
    assert actual[1] == 5.0
    assert actual[2] == 1.0
    assert actual[3] == 5.0

    actual = csyr2k(order=101,
                    uplo=121,
                    trans=111,
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
    assert actual[0] == 9.0
    assert actual[1] == 9.0
    assert actual[2] == 1.0
    assert actual[3] == 9.0


def test_case4(symm_arr_ones_c):
    actual = zsyr2k(order=102,
                    uplo=121,
                    trans=111,
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
    assert actual[0] == 5.0
    assert actual[1] == 1.0
    assert actual[2] == 5.0
    assert actual[3] == 5.0

    actual = zsyr2k(order=102,
                    uplo=122,
                    trans=111,
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
    assert actual[0] == 5.0
    assert actual[1] == 5.0
    assert actual[2] == 1.0
    assert actual[3] == 5.0

    actual = zsyr2k(order=101,
                    uplo=121,
                    trans=111,
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
    assert actual[0] == 9.0
    assert actual[1] == 9.0
    assert actual[2] == 1.0
    assert actual[3] == 9.0
