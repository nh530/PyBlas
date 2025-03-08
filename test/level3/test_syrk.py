import pytest
from pyblas import ssyrk, dsyrk, csyrk, zsyrk


def test_case1(symm_arr_ones):
    actual = ssyrk(order=102,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 1.0
    assert actual[2] == 3.0
    assert actual[3] == 3.0

    actual = ssyrk(order=102,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 1.0
    assert actual[3] == 3.0

    actual = ssyrk(order=101,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 1.0
    assert actual[2] == 3.0
    assert actual[3] == 3.0

    actual = ssyrk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 1.0
    assert actual[3] == 3.0

    actual = ssyrk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=5,
                   c_mat=symm_arr_ones,
                   ldc=2)

    assert actual[0] == 7.0
    assert actual[1] == 7.0
    assert actual[2] == 1.0
    assert actual[3] == 7.0


def test_case2(symm_arr_ones):
    actual = dsyrk(order=102,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 1.0
    assert actual[2] == 3.0
    assert actual[3] == 3.0

    actual = dsyrk(order=102,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 1.0
    assert actual[3] == 3.0

    actual = dsyrk(order=101,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 1.0
    assert actual[2] == 3.0
    assert actual[3] == 3.0

    actual = dsyrk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=1,
                   c_mat=symm_arr_ones,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 1.0
    assert actual[3] == 3.0

    actual = dsyrk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1,
                   a_mat=symm_arr_ones,
                   lda=2,
                   beta=5,
                   c_mat=symm_arr_ones,
                   ldc=2)

    assert actual[0] == 7.0
    assert actual[1] == 7.0
    assert actual[2] == 1.0
    assert actual[3] == 7.0


def test_case3(symm_arr_ones_c):
    actual = csyrk(order=102,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 1.0
    assert actual[2] == 3.0
    assert actual[3] == 3.0

    actual = csyrk(order=102,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 1.0
    assert actual[3] == 3.0

    actual = csyrk(order=101,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 1.0
    assert actual[2] == 3.0
    assert actual[3] == 3.0

    actual = csyrk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 1.0
    assert actual[3] == 3.0

    actual = csyrk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(5.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)

    assert actual[0] == 7.0
    assert actual[1] == 7.0
    assert actual[2] == 1.0
    assert actual[3] == 7.0


def test_case4(symm_arr_ones_c):
    actual = zsyrk(order=102,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 1.0
    assert actual[2] == 3.0
    assert actual[3] == 3.0

    actual = zsyrk(order=102,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 1.0
    assert actual[3] == 3.0

    actual = zsyrk(order=101,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 1.0
    assert actual[2] == 3.0
    assert actual[3] == 3.0

    actual = zsyrk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 3.0
    assert actual[1] == 3.0
    assert actual[2] == 1.0
    assert actual[3] == 3.0

    actual = zsyrk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=symm_arr_ones_c,
                   lda=2,
                   beta=complex(5.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)

    assert actual[0] == 7.0
    assert actual[1] == 7.0
    assert actual[2] == 1.0
    assert actual[3] == 7.0
