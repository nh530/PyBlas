import pytest
from pyblas import strsm, dtrsm, ctrsm, ztrsm


def test_case1(id_2x2_mat, symm_arr_ones):
    actual = strsm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=1,
        a_mat=id_2x2_mat,
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 1.0
    assert actual[1] == 1.0
    assert actual[2] == 1.0
    assert actual[3] == 1.0

    actual = strsm(
        order=102,
        side=142,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=2,
        a_mat=id_2x2_mat,
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 2.0
    assert actual[2] == 2.0
    assert actual[3] == 2.0


def test_case2(id_2x2_mat, symm_arr_ones):
    actual = dtrsm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=1,
        a_mat=id_2x2_mat,
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 1.0
    assert actual[1] == 1.0
    assert actual[2] == 1.0
    assert actual[3] == 1.0

    actual = dtrsm(
        order=102,
        side=142,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=2,
        a_mat=id_2x2_mat,
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 2.0
    assert actual[2] == 2.0
    assert actual[3] == 2.0


def test_case3(id_2x2_mat_c, symm_arr_ones_c):
    actual = ctrsm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=complex(1.0, 0.0),
        a_mat=id_2x2_mat_c,
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 1.0
    assert actual[1] == 1.0
    assert actual[2] == 1.0
    assert actual[3] == 1.0

    actual = ctrsm(
        order=102,
        side=142,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=complex(2.0, 0.0),
        a_mat=id_2x2_mat_c,
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 2.0
    assert actual[2] == 2.0
    assert actual[3] == 2.0


def test_case4(id_2x2_mat_c, symm_arr_ones_c):
    actual = ztrsm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=complex(1.0, 0.0),
        a_mat=id_2x2_mat_c,
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 1.0
    assert actual[1] == 1.0
    assert actual[2] == 1.0
    assert actual[3] == 1.0

    actual = ztrsm(
        order=102,
        side=142,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=complex(2.0, 0.0),
        a_mat=id_2x2_mat_c,
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 2.0
    assert actual[2] == 2.0
    assert actual[3] == 2.0
