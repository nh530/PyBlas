import pytest


@pytest.fixture
def symm_arr_ones():
    """
    symmetric ones array  2x2
    """
    return [1.0, 1.0, 1.0, 1.0]


@pytest.fixture
def symm_arr():
    return [1.0, 1.0, 2.0, 2.0]


@pytest.fixture
def sym_arr_ones_lda3():
    """ 3x3 array
    """
    return [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def arr_3x2_ones_lda3():
    return [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]


@pytest.fixture
def id_2x2_mat():
    # cblas column oriented
    return [1.0, 0.0, 0.0, 1.0]


@pytest.fixture
def id_2x2_mat_c():
    # cblas column oriented
    return [
        complex(1.0, 0.0),
        complex(0.0, 0.0),
        complex(0.0, 0.0),
        complex(1.0, 0.0)
    ]


@pytest.fixture
def symm_arr_ones_c():
    """
    complex symmetric ones array  2x2
    """
    return [
        complex(1.0, 0.0),
        complex(1.0, 0.0),
        complex(1.0, 0.0),
        complex(1.0, 0.0)
    ]
