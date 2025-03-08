from enum import Enum


class Layout(Enum):
    """
    Specifies whether two-dimensional array storage is row-major (CblasRowMajor) or column-major (CblasColMajor).
    """
    ROW = 101
    COL = 102


class Transpose(Enum):
    """
    Determines if the matrix operation should be transposed.
    """
    NO_TRANS = 111
    TRANS = 112
    CONJ_TRANS = 113


class Triangular(Enum):
    """
    Specifies whether the upper or lower triangular part of the array is used.
    If Upper, then the upper triangular of the array c is used.
    If Lower, then the low triangular of the array c is used.
    """
    UPPER = 121
    LOWER = 122


class Diagonal(Enum):
    """
    Determines if unit triangular matrix.
    """
    NON_UNIT = 131
    UNIT = 132


class Side(Enum):
    """
    Specifies whether op(A) appears on the left or right of X in the equation:
    """
    LEFT = 141
    RIGHT = 142
