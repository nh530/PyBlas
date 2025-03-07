import pytest
from pyblas import chemm, zhemm


def test_case1(id_2x2_mat_c, symm_arr_ones_c):
    actual = chemm(order=101,
                   side=141,
                   uplo=121,
                   m=2,
                   n=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   lda=2,
                   b_mat=id_2x2_mat_c,
                   ldb=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 2.0
    assert actual[1] == complex(5.0, 1.0)
    assert actual[2] == complex(5.0, -1.0)
    assert actual[3] == 2.0

    actual = chemm(order=101,
                   side=142,
                   uplo=122,
                   m=2,
                   n=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   lda=2,
                   b_mat=id_2x2_mat_c,
                   ldb=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 2.0
    assert actual[1] == complex(5.0, 1.0)
    assert actual[2] == complex(5.0, -1.0)
    assert actual[3] == 2.0


    actual = chemm(order=101,
                   side=141,
                   uplo=121,
                   m=2,
                   n=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   lda=2,
                   b_mat=id_2x2_mat_c,
                   ldb=2,
                   beta=complex(5.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 6.0
    assert actual[1] == complex(9.0, 1.0)
    assert actual[2] == complex(9.0, -1.0)
    assert actual[3] == 6.0


def test_case2(id_2x2_mat_c, symm_arr_ones_c):
    actual = zhemm(order=101,
                   side=141,
                   uplo=121,
                   m=2,
                   n=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   lda=2,
                   b_mat=id_2x2_mat_c,
                   ldb=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 2.0
    assert actual[1] == complex(5.0, 1.0)
    assert actual[2] == complex(5.0, -1.0)
    assert actual[3] == 2.0

    actual = zhemm(order=101,
                   side=142,
                   uplo=122,
                   m=2,
                   n=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   lda=2,
                   b_mat=id_2x2_mat_c,
                   ldb=2,
                   beta=complex(1.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 2.0
    assert actual[1] == complex(5.0, 1.0)
    assert actual[2] == complex(5.0, -1.0)
    assert actual[3] == 2.0


    actual = zhemm(order=101,
                   side=141,
                   uplo=121,
                   m=2,
                   n=2,
                   alpha=complex(1.0, 0.0),
                   a_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   lda=2,
                   b_mat=id_2x2_mat_c,
                   ldb=2,
                   beta=complex(5.0, 0.0),
                   c_mat=symm_arr_ones_c,
                   ldc=2)
    assert actual[0] == 6.0
    assert actual[1] == complex(9.0, 1.0)
    assert actual[2] == complex(9.0, -1.0)
    assert actual[3] == 6.0

