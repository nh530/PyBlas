import pytest
from pyblas import cherk, zherk


def test_case1(id_2x2_mat_c, symm_arr_ones_c):
    actual = cherk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1.0,
                   a_mat=id_2x2_mat_c,
                   lda=2,
                   beta=1.0,
                   c_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   ldc=2)
    assert actual[0] == 2.0
    assert actual[1] == complex(4.0, 1.0)
    assert actual[2] == complex(4.0, -1.0)
    assert actual[3] == 2.0

    actual = cherk(order=101,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1.0,
                   a_mat=id_2x2_mat_c,
                   lda=2,
                   beta=1.0,
                   c_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   ldc=2)
    assert actual[0] == 2.0
    assert actual[1] == complex(4.0, 1.0)
    assert actual[2] == complex(4.0, -1.0)
    assert actual[3] == 2.0

    actual = cherk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1.0,
                   a_mat=id_2x2_mat_c,
                   lda=2,
                   beta=5.0,
                   c_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   ldc=2)
    assert actual[0] == 6.0
    assert actual[1] == complex(20.0, 5.0)
    assert actual[2] == complex(4.0, -1.0)
    assert actual[3] == 6.0

    actual = cherk(order=101,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1.0,
                   a_mat=id_2x2_mat_c,
                   lda=2,
                   beta=5.0,
                   c_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   ldc=2)
    assert actual[0] == 6.0
    assert actual[1] == complex(4.0, 1.0)
    assert actual[2] == complex(20.0, -5.0)
    assert actual[3] == 6.0



def test_case2(id_2x2_mat_c, symm_arr_ones_c):
    actual = zherk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1.0,
                   a_mat=id_2x2_mat_c,
                   lda=2,
                   beta=1.0,
                   c_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   ldc=2)
    assert actual[0] == 2.0
    assert actual[1] == complex(4.0, 1.0)
    assert actual[2] == complex(4.0, -1.0)
    assert actual[3] == 2.0

    actual = zherk(order=101,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1.0,
                   a_mat=id_2x2_mat_c,
                   lda=2,
                   beta=1.0,
                   c_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   ldc=2)
    assert actual[0] == 2.0
    assert actual[1] == complex(4.0, 1.0)
    assert actual[2] == complex(4.0, -1.0)
    assert actual[3] == 2.0

    actual = zherk(order=101,
                   uplo=121,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1.0,
                   a_mat=id_2x2_mat_c,
                   lda=2,
                   beta=5.0,
                   c_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   ldc=2)
    assert actual[0] == 6.0
    assert actual[1] == complex(20.0, 5.0)
    assert actual[2] == complex(4.0, -1.0)
    assert actual[3] == 6.0

    actual = zherk(order=101,
                   uplo=122,
                   trans=111,
                   n=2,
                   k=2,
                   alpha=1.0,
                   a_mat=id_2x2_mat_c,
                   lda=2,
                   beta=5.0,
                   c_mat=[
                       complex(1.0, 0.0),
                       complex(4.0, 1.0),
                       complex(4.0, -1.0),
                       complex(1.0, 0.0)
                   ],
                   ldc=2)
    assert actual[0] == 6.0
    assert actual[1] == complex(4.0, 1.0)
    assert actual[2] == complex(20.0, -5.0)
    assert actual[3] == 6.0
