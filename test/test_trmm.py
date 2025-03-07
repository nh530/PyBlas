import pytest
from pyblas import strmm, dtrmm, ctrmm, ztrmm


def test_case1(symm_arr_ones):
    actual = strmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=1,
        a_mat=symm_arr_ones,
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 1.0
    assert actual[2] == 2.0
    assert actual[3] == 1.0

    actual = strmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=1,
        a_mat=[1.0, 0.0, 1.0, 1.0],
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 1.0
    assert actual[2] == 2.0
    assert actual[3] == 1.0

    actual = strmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=132,
        m=2,
        n=2,
        alpha=2,
        a_mat=[1, 0, 0, 1],
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 2.0
    assert actual[2] == 2.0
    assert actual[3] == 2.0


def test_case2(symm_arr_ones):
    actual = dtrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=1,
        a_mat=symm_arr_ones,
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 1.0
    assert actual[2] == 2.0
    assert actual[3] == 1.0

    actual = dtrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=1,
        a_mat=[1.0, 0.0, 1.0, 1.0],
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 1.0
    assert actual[2] == 2.0
    assert actual[3] == 1.0

    actual = dtrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=132,
        m=2,
        n=2,
        alpha=2,
        a_mat=[1, 0, 0, 1],
        lda=2,
        b_mat=symm_arr_ones,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 2.0
    assert actual[2] == 2.0
    assert actual[3] == 2.0


def test_case3(symm_arr_ones_c):
    actual = ctrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=complex(1.0, 0.0),
        a_mat=symm_arr_ones_c,
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 1.0
    assert actual[2] == 2.0
    assert actual[3] == 1.0

    actual = ctrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=complex(1.0, 0.0),
        a_mat=[complex(1.0, 0.0), complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0)],
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 1.0
    assert actual[2] == 2.0
    assert actual[3] == 1.0

    actual = ctrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=132,
        m=2,
        n=2,
        alpha=complex(2.0, 0.0),
        a_mat=[complex(1.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(1.0, 0.0)],
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 2.0
    assert actual[2] == 2.0
    assert actual[3] == 2.0


def test_case4(symm_arr_ones_c):
    actual = ztrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=complex(1.0, 0.0),
        a_mat=symm_arr_ones_c,
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 1.0
    assert actual[2] == 2.0
    assert actual[3] == 1.0

    actual = ztrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=131,
        m=2,
        n=2,
        alpha=complex(1.0, 0.0),
        a_mat=[complex(1.0, 0.0), complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0)],
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 1.0
    assert actual[2] == 2.0
    assert actual[3] == 1.0

    actual = ztrmm(
        order=102,
        side=141,
        uplo=121,
        trans=111,
        diag=132,
        m=2,
        n=2,
        alpha=complex(2.0, 0.0),
        a_mat=[complex(1.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(1.0, 0.0)],
        lda=2,
        b_mat=symm_arr_ones_c,
        ldb=2,
    )
    assert actual[0] == 2.0
    assert actual[1] == 2.0
    assert actual[2] == 2.0
    assert actual[3] == 2.0
