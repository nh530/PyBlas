#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cblas.h>
#include <complex>
#include <iostream>
// TODO: Can probably optimize the for loops for symmetric matrices

extern "C" {
static PyObject *CallError;

static PyObject *sgemm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {
      (char *)"order", (char *)"transa", (char *)"transb", (char *)"m",   (char *)"n",
      (char *)"k",     (char *)"alpha",  (char *)"a_mat",  (char *)"lda", (char *)"b_mat",
      (char *)"ldb",   (char *)"beta",   (char *)"c_mat",  (char *)"ldc", NULL}; // Array of char pointers; cast to char * for c style string literal
                                                                                 // to c string conversion
  ::CBLAS_ORDER order;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_TRANSPOSE transb;
  int m;
  int n;
  int k;
  float alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  float beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiifOiOifOi", kwlist, &order, &transa, &transb, &m, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb,
                                   &beta, &c_mat, &ldc)) {
    return NULL;
  }
  float *amat = new float[PyList_Size((PyObject *)a_mat)];
  float *bmat = new float[PyList_Size((PyObject *)b_mat)];
  float *cmat = new float[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    cmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)c_mat, i));
  }

  ::cblas_sgemm(order, transa, transb, m, n, k, alpha, amat, lda, bmat, ldb, beta, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(cmat[i]));
  }
  delete[] amat;
  delete[] bmat;
  delete[] cmat;
  return out;
}

static PyObject *ssymm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {
      (char *)"order", (char *)"side", (char *)"uplo",  (char *)"m",   (char *)"n", (char *)"alpha", (char *)"a_mat", (char *)"lda", (char *)"b_mat",
      (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc", NULL}; // Array of char pointers; cast to char * for c style string literal
  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  int m;
  int n;
  float alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  float beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiifOiOifOi", kwlist, &order, &side, &uplo, &m, &n, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  float *amat = new float[PyList_Size((PyObject *)a_mat)];
  float *bmat = new float[PyList_Size((PyObject *)b_mat)];
  float *cmat = new float[PyList_Size((PyObject *)c_mat)];

  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    cmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)c_mat, i));
  }

  ::cblas_ssymm(order, side, uplo, m, n, alpha, amat, lda, bmat, ldb, beta, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(cmat[i]));
  }

  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}

static PyObject *ssyrk(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo", (char *)"trans", (char *)"n",     (char *)"k",   (char *)"alpha",
                    (char *)"a_mat", (char *)"lda",  (char *)"beta",  (char *)"c_mat", (char *)"ldc", NULL};
  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  float alpha;
  PyListObject *a_mat;
  int lda;
  float beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiifOifOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &beta, &c_mat, &ldc)) {
    return NULL;
  }
  float *amat = new float[PyList_Size((PyObject *)a_mat)];
  float *cmat = new float[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    cmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)c_mat, i));
  }
  ::cblas_ssyrk(order, uplo, trans, n, k, alpha, amat, lda, beta, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(cmat[i]));
  }

  delete[] amat;
  delete[] cmat;

  return out;
}
static PyObject *ssyr2k(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo",  (char *)"trans", (char *)"n",    (char *)"k",     (char *)"alpha", (char *)"a_mat",
                    (char *)"lda",   (char *)"b_mat", (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc",   NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  float alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  float beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiifOiOifOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  float *amat = new float[PyList_Size((PyObject *)a_mat)];
  float *bmat = new float[PyList_Size((PyObject *)b_mat)];
  float *cmat = new float[PyList_Size((PyObject *)c_mat)];

  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    cmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)c_mat, i));
  }
  ::cblas_ssyr2k(order, uplo, trans, n, k, alpha, amat, lda, bmat, ldb, beta, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(cmat[i]));
  }
  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *strmm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"trans", (char *)"diag", (char *)"m", (char *)"n",
                    (char *)"alpha", (char *)"a_mat", (char *)"lda",  (char *)"b_mat", (char *)"ldb",  NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_DIAG diag;
  int m;
  int n;
  float alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiifOiOi", kwlist, &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a_mat, &lda, &b_mat,
                                   &ldb)) {
    return NULL;
  }
  float *amat = new float[PyList_Size((PyObject *)a_mat)];
  float *bmat = new float[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }

  ::cblas_strmm(order, side, uplo, transa, diag, m, n, alpha, amat, lda, bmat, ldb);
  PyObject *out = PyList_New(PyList_Size((PyObject *)b_mat));
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(bmat[i]));
  }
  delete[] amat;
  delete[] bmat;
  return out;
}
static PyObject *strsm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"trans", (char *)"diag", (char *)"m", (char *)"n",
                    (char *)"alpha", (char *)"a_mat", (char *)"lda",  (char *)"b_mat", (char *)"ldb",  NULL};
  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_DIAG diag;
  int m;
  int n;
  float alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiifOiOi", kwlist, &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a_mat, &lda, &b_mat,
                                   &ldb)) {
    return NULL;
  }
  float *amat = new float[PyList_Size((PyObject *)a_mat)];
  float *bmat = new float[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = (float)PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }

  ::cblas_strsm(order, side, uplo, transa, diag, m, n, alpha, amat, lda, bmat, ldb);
  PyObject *out = PyList_New(PyList_Size((PyObject *)b_mat));
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(bmat[i]));
  }

  delete[] amat;
  delete[] bmat;
  return out;
}
static PyObject *dgemm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"transa", (char *)"transb", (char *)"m",   (char *)"n",
                    (char *)"k",     (char *)"alpha",  (char *)"a_mat",  (char *)"lda", (char *)"b_mat",
                    (char *)"ldb",   (char *)"beta",   (char *)"c_mat",  (char *)"ldc", NULL};
  CBLAS_ORDER order;
  CBLAS_TRANSPOSE transa;
  CBLAS_TRANSPOSE transb;
  int m;
  int n;
  int k;
  double alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  double beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiidOiOidOi", kwlist, &order, &transa, &transb, &m, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb,
                                   &beta, &c_mat, &ldc)) {
    return NULL;
  }
  double *amat = new double[PyList_Size((PyObject *)a_mat)];
  double *bmat = new double[PyList_Size((PyObject *)b_mat)];
  double *cmat = new double[PyList_Size((PyObject *)c_mat)];

  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    cmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)c_mat, i));
  }

  ::cblas_dgemm(order, transa, transb, m, n, k, alpha, amat, lda, bmat, ldb, beta, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(cmat[i]));
  }
  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *dsymm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {
      (char *)"order", (char *)"side", (char *)"uplo",  (char *)"m",   (char *)"n", (char *)"alpha", (char *)"a_mat", (char *)"lda", (char *)"b_mat",
      (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc", NULL}; // Array of char pointers; cast to char * for c style string literal
                                                                              // to c string conversion
  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  int m;
  int n;
  double alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  double beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiidOiOidOi", kwlist, &order, &side, &uplo, &m, &n, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  double *amat = new double[PyList_Size((PyObject *)a_mat)];
  double *bmat = new double[PyList_Size((PyObject *)b_mat)];
  double *cmat = new double[PyList_Size((PyObject *)c_mat)];

  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    cmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)c_mat, i));
  }

  ::cblas_dsymm(order, side, uplo, m, n, alpha, amat, lda, bmat, ldb, beta, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(cmat[i]));
  }
  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *dsyrk(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo", (char *)"trans", (char *)"n",     (char *)"k",   (char *)"alpha",
                    (char *)"a_mat", (char *)"lda",  (char *)"beta",  (char *)"c_mat", (char *)"ldc", NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  double alpha;
  PyListObject *a_mat;
  int lda;
  double beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiidOidOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &beta, &c_mat, &ldc)) {
    return NULL;
  }
  double *amat = new double[PyList_Size((PyObject *)a_mat)];
  double *cmat = new double[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    cmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)c_mat, i));
  }

  ::cblas_dsyrk(order, uplo, trans, n, k, alpha, amat, lda, beta, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(cmat[i]));
  }
  delete[] amat;
  delete[] cmat;

  return out;
}
static PyObject *dsyr2k(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo",  (char *)"trans", (char *)"n",    (char *)"k",     (char *)"alpha", (char *)"a_mat",
                    (char *)"lda",   (char *)"b_mat", (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc",   NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  double alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  double beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiidOiOidOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  double *amat = new double[PyList_Size((PyObject *)a_mat)];
  double *bmat = new double[PyList_Size((PyObject *)b_mat)];
  double *cmat = new double[PyList_Size((PyObject *)c_mat)];

  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    cmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)c_mat, i));
  }
  ::cblas_dsyr2k(order, uplo, trans, n, k, alpha, amat, lda, bmat, ldb, beta, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(cmat[i]));
  }
  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *dtrmm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"trans", (char *)"diag", (char *)"m", (char *)"n",
                    (char *)"alpha", (char *)"a_mat", (char *)"lda",  (char *)"b_mat", (char *)"ldb",  NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_DIAG diag;
  int m;
  int n;
  double alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiidOiOi", kwlist, &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a_mat, &lda, &b_mat,
                                   &ldb)) {
    return NULL;
  }
  double *amat = new double[PyList_Size((PyObject *)a_mat)];
  double *bmat = new double[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }

  ::cblas_dtrmm(order, side, uplo, transa, diag, m, n, alpha, amat, lda, bmat, ldb);
  PyObject *out = PyList_New(PyList_Size((PyObject *)b_mat));
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(bmat[i]));
  }
  delete[] amat;
  delete[] bmat;

  return out;
}
static PyObject *dtrsm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"trans", (char *)"diag", (char *)"m", (char *)"n",
                    (char *)"alpha", (char *)"a_mat", (char *)"lda",  (char *)"b_mat", (char *)"ldb",  NULL};
  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_DIAG diag;
  int m;
  int n;
  double alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiidOiOi", kwlist, &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a_mat, &lda, &b_mat,
                                   &ldb)) {
    return NULL;
  }
  double *amat = new double[PyList_Size((PyObject *)a_mat)];
  double *bmat = new double[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    amat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)a_mat, i));
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    bmat[i] = PyFloat_AS_DOUBLE(PyList_GetItem((PyObject *)b_mat, i));
  }

  ::cblas_dtrsm(order, side, uplo, transa, diag, m, n, alpha, amat, lda, bmat, ldb);
  PyObject *out = PyList_New(PyList_Size((PyObject *)b_mat));
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    PyList_SET_ITEM(out, i, PyFloat_FromDouble(bmat[i]));
  }
  delete[] amat;
  delete[] bmat;

  return out;
}
static PyObject *cgemm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {
      (char *)"order", (char *)"transa", (char *)"transb", (char *)"m",   (char *)"n",
      (char *)"k",     (char *)"alpha",  (char *)"a_mat",  (char *)"lda", (char *)"b_mat",
      (char *)"ldb",   (char *)"beta",   (char *)"c_mat",  (char *)"ldc", NULL}; // Array of char pointers; cast to char * for c style string literal
                                                                                 // to c string conversion
  ::CBLAS_ORDER order;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_TRANSPOSE transb;
  int m;
  int n;
  int k;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiDOiOiDOi", kwlist, &order, &transa, &transb, &m, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb,
                                   &beta, &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *bmat = new std::complex<float>[PyList_Size((PyObject *)b_mat)];
  std::complex<float> *cmat = new std::complex<float>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<float>(temp.real, temp.imag);
  }

  std::complex<float> alpha_c(alpha.real, alpha.imag);
  std::complex<float> beta_c(beta.real, beta.imag);
  ::cblas_cgemm(order, transa, transb, m, n, k, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb, (void *)&beta_c, (void *)cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }
  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *csymm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {
      (char *)"order", (char *)"side", (char *)"uplo",  (char *)"m",   (char *)"n", (char *)"alpha", (char *)"a_mat", (char *)"lda", (char *)"b_mat",
      (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc", NULL}; // Array of char pointers; cast to char * for c style string literal
                                                                              // to c string conversion
  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  int m;
  int n;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiOiDOi", kwlist, &order, &side, &uplo, &m, &n, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *bmat = new std::complex<float>[PyList_Size((PyObject *)b_mat)];
  std::complex<float> *cmat = new std::complex<float>[PyList_Size((PyObject *)c_mat)];

  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<float>(temp.real, temp.imag);
  }

  std::complex<float> alpha_c(alpha.real, alpha.imag);
  std::complex<float> beta_c(beta.real, beta.imag);
  ::cblas_csymm(order, side, uplo, m, n, (void *)&alpha_c, amat, lda, bmat, ldb, (void *)&beta_c, cmat, ldc);
  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *csyrk(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo", (char *)"trans", (char *)"n",     (char *)"k",   (char *)"alpha",
                    (char *)"a_mat", (char *)"lda",  (char *)"beta",  (char *)"c_mat", (char *)"ldc", NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiDOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &beta, &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *cmat = new std::complex<float>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  std::complex<float> alpha_c(alpha.real, alpha.imag);
  std::complex<float> beta_c(beta.real, beta.imag);
  ::cblas_csyrk(order, uplo, trans, n, k, (void *)&alpha_c, (void *)amat, lda, (void *)&beta_c, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] cmat;

  return out;
}
static PyObject *csyr2k(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo",  (char *)"trans", (char *)"n",    (char *)"k",     (char *)"alpha", (char *)"a_mat",
                    (char *)"lda",   (char *)"b_mat", (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc",   NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiOiDOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *bmat = new std::complex<float>[PyList_Size((PyObject *)b_mat)];
  std::complex<float> *cmat = new std::complex<float>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<float>(temp.real, temp.imag);
  }

  std::complex<float> alpha_c(alpha.real, alpha.imag);
  std::complex<float> beta_c(beta.real, beta.imag);
  ::cblas_csyr2k(order, uplo, trans, n, k, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb, (void *)&beta_c, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *ctrmm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"trans", (char *)"diag", (char *)"m", (char *)"n",
                    (char *)"alpha", (char *)"a_mat", (char *)"lda",  (char *)"b_mat", (char *)"ldb",  NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_DIAG diag;
  int m;
  int n;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiiDOiOi", kwlist, &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a_mat, &lda, &b_mat,
                                   &ldb)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *bmat = new std::complex<float>[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  std::complex<float> alpha_c(alpha.real, alpha.imag);
  ::cblas_ctrmm(order, side, uplo, transa, diag, m, n, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb);

  PyObject *out = PyList_New(PyList_Size((PyObject *)b_mat));
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(bmat[i].real(), bmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;

  return out;
}
static PyObject *ctrsm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"trans", (char *)"diag", (char *)"m", (char *)"n",
                    (char *)"alpha", (char *)"a_mat", (char *)"lda",  (char *)"b_mat", (char *)"ldb",  NULL};
  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_DIAG diag;
  int m;
  int n;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiiDOiOi", kwlist, &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a_mat, &lda, &b_mat,
                                   &ldb)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *bmat = new std::complex<float>[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  std::complex<float> alpha_c(alpha.real, alpha.imag);
  ::cblas_ctrsm(order, side, uplo, transa, diag, m, n, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb);

  PyObject *out = PyList_New(PyList_Size((PyObject *)b_mat));
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(bmat[i].real(), bmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;

  return out;
}
static PyObject *zgemm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {
      (char *)"order", (char *)"transa", (char *)"transb", (char *)"m",   (char *)"n",
      (char *)"k",     (char *)"alpha",  (char *)"a_mat",  (char *)"lda", (char *)"b_mat",
      (char *)"ldb",   (char *)"beta",   (char *)"c_mat",  (char *)"ldc", NULL}; // Array of char pointers; cast to char * for c style string literal
                                                                                 // to c string conversion
  ::CBLAS_ORDER order;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_TRANSPOSE transb;
  int m;
  int n;
  int k;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiDOiOiDOi", kwlist, &order, &transa, &transb, &m, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb,
                                   &beta, &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *bmat = new std::complex<double>[PyList_Size((PyObject *)b_mat)];
  std::complex<double> *cmat = new std::complex<double>[PyList_Size((PyObject *)c_mat)];

  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<double>(temp.real, temp.imag);
  }

  std::complex<double> alpha_c(alpha.real, alpha.imag);
  std::complex<double> beta_c(beta.real, beta.imag);
  ::cblas_zgemm(order, transa, transb, m, n, k, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb, (void *)&beta_c, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }
  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *zsymm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {
      (char *)"order", (char *)"side", (char *)"uplo",  (char *)"m",   (char *)"n", (char *)"alpha", (char *)"a_mat", (char *)"lda", (char *)"b_mat",
      (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc", NULL}; // Array of char pointers; cast to char * for c style string literal
                                                                              // to c string conversion
  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  int m;
  int n;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiOiDOi", kwlist, &order, &side, &uplo, &m, &n, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *bmat = new std::complex<double>[PyList_Size((PyObject *)b_mat)];
  std::complex<double> *cmat = new std::complex<double>[PyList_Size((PyObject *)c_mat)];

  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<double>(temp.real, temp.imag);
  }

  std::complex<double> alpha_c(alpha.real, alpha.imag);
  std::complex<double> beta_c(beta.real, beta.imag);
  ::cblas_zsymm(order, side, uplo, m, n, (void *)&alpha_c, amat, lda, bmat, ldb, (void *)&beta_c, cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *zsyrk(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo", (char *)"trans", (char *)"n",     (char *)"k",   (char *)"alpha",
                    (char *)"a_mat", (char *)"lda",  (char *)"beta",  (char *)"c_mat", (char *)"ldc", NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiDOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &beta, &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *cmat = new std::complex<double>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<double>(temp.real, temp.imag);
  }

  std::complex<double> alpha_c(alpha.real, alpha.imag);
  std::complex<double> beta_c(beta.real, beta.imag);
  ::cblas_zsyrk(order, uplo, trans, n, k, (void *)&alpha_c, (void *)amat, lda, (void *)&beta_c, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] cmat;

  return out;
}
static PyObject *zsyr2k(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo",  (char *)"trans", (char *)"n",    (char *)"k",     (char *)"alpha", (char *)"a_mat",
                    (char *)"lda",   (char *)"b_mat", (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc",   NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiOiDOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *bmat = new std::complex<double>[PyList_Size((PyObject *)b_mat)];
  std::complex<double> *cmat = new std::complex<double>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  std::complex<double> alpha_c(alpha.real, alpha.imag);
  std::complex<double> beta_c(beta.real, beta.imag);
  ::cblas_zsyr2k(order, uplo, trans, n, k, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb, (void *)&beta_c, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] cmat;

  return out;
}
static PyObject *ztrmm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"trans", (char *)"diag", (char *)"m", (char *)"n",
                    (char *)"alpha", (char *)"a_mat", (char *)"lda",  (char *)"b_mat", (char *)"ldb",  NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_DIAG diag;
  int m;
  int n;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiiDOiOi", kwlist, &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a_mat, &lda, &b_mat,
                                   &ldb)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *bmat = new std::complex<double>[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  std::complex<double> alpha_c(alpha.real, alpha.imag);
  ::cblas_ztrmm(order, side, uplo, transa, diag, m, n, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb);

  PyObject *out = PyList_New(PyList_Size((PyObject *)b_mat));
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(bmat[i].real(), bmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;

  return out;
}
static PyObject *ztrsm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"trans", (char *)"diag", (char *)"m", (char *)"n",
                    (char *)"alpha", (char *)"a_mat", (char *)"lda",  (char *)"b_mat", (char *)"ldb",  NULL};
  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE transa;
  ::CBLAS_DIAG diag;
  int m;
  int n;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiiiDOiOi", kwlist, &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a_mat, &lda, &b_mat,
                                   &ldb)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *bmat = new std::complex<double>[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  std::complex<double> alpha_c(alpha.real, alpha.imag);
  ::cblas_ztrsm(order, side, uplo, transa, diag, m, n, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb);

  PyObject *out = PyList_New(PyList_Size((PyObject *)b_mat));
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(bmat[i].real(), bmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;

  return out;
}
static PyObject *chemm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"m",    (char *)"n",     (char *)"alpha", (char *)"a_mat",
                    (char *)"lda",   (char *)"b_mat", (char *)"ldb",  (char *)"beta", (char *)"c_mat", (char *)"ldc",   NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  int m;
  int n;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiOiDOi", kwlist, &order, &side, &uplo, &m, &n, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *bmat = new std::complex<float>[PyList_Size((PyObject *)b_mat)];
  std::complex<float> *cmat = new std::complex<float>[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<double>(temp.real, temp.imag);
  }

  std::complex<float> alpha_c(alpha.real, alpha.imag);
  std::complex<float> beta_c(beta.real, beta.imag);
  ::cblas_chemm(order, side, uplo, m, n, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb, (void *)&beta_c, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *cherk(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo", (char *)"trans", (char *)"n",     (char *)"k",   (char *)"alpha",
                    (char *)"a_mat", (char *)"lda",  (char *)"beta",  (char *)"c_mat", (char *)"ldc", NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  float alpha;
  PyListObject *a_mat;
  int lda;
  float beta;
  PyListObject *c_mat;
  int ldc;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiifOifOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &beta, &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *cmat = new std::complex<float>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  ::cblas_cherk(order, uplo, trans, n, k, alpha, amat, lda, beta, cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] cmat;

  return out;
}
static PyObject *cher2k(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo",  (char *)"trans", (char *)"n",    (char *)"k",     (char *)"alpha", (char *)"a_mat",
                    (char *)"lda",   (char *)"b_mat", (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc",   NULL};
  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  float beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiOifOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<float> *amat = new std::complex<float>[PyList_Size((PyObject *)a_mat)];
  std::complex<float> *bmat = new std::complex<float>[PyList_Size((PyObject *)b_mat)];
  std::complex<float> *cmat = new std::complex<float>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<float>(temp.real, temp.imag);
  }
  std::complex<float> alpha_c(alpha.real, alpha.imag);
  ::cblas_cher2k(order, uplo, trans, n, k, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb, beta, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *zhemm(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"side",  (char *)"uplo", (char *)"m",    (char *)"n",     (char *)"alpha", (char *)"a_mat",
                    (char *)"lda",   (char *)"b_mat", (char *)"ldb",  (char *)"beta", (char *)"c_mat", (char *)"ldc",   NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_SIDE side;
  ::CBLAS_UPLO uplo;
  int m;
  int n;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  Py_complex beta;
  PyListObject *c_mat;
  int ldc;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiOiDOi", kwlist, &order, &side, &uplo, &m, &n, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *bmat = new std::complex<double>[PyList_Size((PyObject *)b_mat)];
  std::complex<double> *cmat = new std::complex<double>[PyList_Size((PyObject *)b_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<double>(temp.real, temp.imag);
  }

  std::complex<double> alpha_c(alpha.real, alpha.imag);
  std::complex<double> beta_c(beta.real, beta.imag);
  ::cblas_zhemm(order, side, uplo, m, n, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb, (void *)&beta_c, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}
static PyObject *zherk(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo", (char *)"trans", (char *)"n",     (char *)"k",   (char *)"alpha",
                    (char *)"a_mat", (char *)"lda",  (char *)"beta",  (char *)"c_mat", (char *)"ldc", NULL};

  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  float alpha;
  PyListObject *a_mat;
  int lda;
  float beta;
  PyListObject *c_mat;
  int ldc;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiifOifOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &beta, &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *cmat = new std::complex<double>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  ::cblas_zherk(order, uplo, trans, n, k, alpha, amat, lda, beta, cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] cmat;

  return out;
}
static PyObject *zher2k(PyObject *self, PyObject *args, PyObject *kwds) {
  char *kwlist[] = {(char *)"order", (char *)"uplo",  (char *)"trans", (char *)"n",    (char *)"k",     (char *)"alpha", (char *)"a_mat",
                    (char *)"lda",   (char *)"b_mat", (char *)"ldb",   (char *)"beta", (char *)"c_mat", (char *)"ldc",   NULL};
  ::CBLAS_ORDER order;
  ::CBLAS_UPLO uplo;
  ::CBLAS_TRANSPOSE trans;
  int n;
  int k;
  Py_complex alpha;
  PyListObject *a_mat;
  int lda;
  PyListObject *b_mat;
  int ldb;
  float beta;
  PyListObject *c_mat;
  int ldc;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiDOiOifOi", kwlist, &order, &uplo, &trans, &n, &k, &alpha, &a_mat, &lda, &b_mat, &ldb, &beta,
                                   &c_mat, &ldc)) {
    return NULL;
  }
  std::complex<double> *amat = new std::complex<double>[PyList_Size((PyObject *)a_mat)];
  std::complex<double> *bmat = new std::complex<double>[PyList_Size((PyObject *)b_mat)];
  std::complex<double> *cmat = new std::complex<double>[PyList_Size((PyObject *)c_mat)];
  for (int i = 0; i < PyList_Size((PyObject *)a_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)a_mat, i));
    amat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)b_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)b_mat, i));
    bmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    Py_complex temp = PyComplex_AsCComplex(PyList_GetItem((PyObject *)c_mat, i));
    cmat[i] = std::complex<double>(temp.real, temp.imag);
  }
  std::complex<double> alpha_c(alpha.real, alpha.imag);
  ::cblas_zher2k(order, uplo, trans, n, k, (void *)&alpha_c, (void *)amat, lda, (void *)bmat, ldb, beta, (void *)cmat, ldc);

  PyObject *out = PyList_New(PyList_Size((PyObject *)c_mat));
  for (int i = 0; i < PyList_Size((PyObject *)c_mat); i++) {
    PyList_SET_ITEM(out, i, PyComplex_FromDoubles(cmat[i].real(), cmat[i].imag()));
  }

  delete[] amat;
  delete[] bmat;
  delete[] cmat;

  return out;
}

static PyMethodDef cblasmethods[] = {
    {"sgemm", (PyCFunction)sgemm, METH_VARARGS | METH_KEYWORDS, ""},
    {"ssymm", (PyCFunction)ssymm, METH_VARARGS | METH_KEYWORDS, ""},
    {"ssyrk", (PyCFunction)ssyrk, METH_VARARGS | METH_KEYWORDS, ""},
    {"ssyr2k", (PyCFunction)ssyr2k, METH_VARARGS | METH_KEYWORDS, ""},
    {"strmm", (PyCFunction)strmm, METH_VARARGS | METH_KEYWORDS, ""},
    {"strsm", (PyCFunction)strsm, METH_VARARGS | METH_KEYWORDS, ""},
    {"dgemm", (PyCFunction)dgemm, METH_VARARGS | METH_KEYWORDS, ""},
    {"dsymm", (PyCFunction)dsymm, METH_VARARGS | METH_KEYWORDS, ""},
    {"dsyrk", (PyCFunction)dsyrk, METH_VARARGS | METH_KEYWORDS, ""},
    {"dsyr2k", (PyCFunction)dsyr2k, METH_VARARGS | METH_KEYWORDS, ""},
    {"dtrmm", (PyCFunction)dtrmm, METH_VARARGS | METH_KEYWORDS, ""},
    {"dtrsm", (PyCFunction)dtrsm, METH_VARARGS | METH_KEYWORDS, ""},
    {"cgemm", (PyCFunction)cgemm, METH_VARARGS | METH_KEYWORDS, ""},
    {"csymm", (PyCFunction)csymm, METH_VARARGS | METH_KEYWORDS, ""},
    {"csyrk", (PyCFunction)csyrk, METH_VARARGS | METH_KEYWORDS, ""},
    {"csyr2k", (PyCFunction)csyr2k, METH_VARARGS | METH_KEYWORDS, ""},
    {"ctrmm", (PyCFunction)ctrmm, METH_VARARGS | METH_KEYWORDS, ""},
    {"ctrsm", (PyCFunction)ctrsm, METH_VARARGS | METH_KEYWORDS, ""},
    {"zgemm", (PyCFunction)zgemm, METH_VARARGS | METH_KEYWORDS, ""},
    {"zsymm", (PyCFunction)zsymm, METH_VARARGS | METH_KEYWORDS, ""},
    {"zsyrk", (PyCFunction)zsyrk, METH_VARARGS | METH_KEYWORDS, ""},
    {"zsyr2k", (PyCFunction)zsyr2k, METH_VARARGS | METH_KEYWORDS, ""},
    {"ztrmm", (PyCFunction)ztrmm, METH_VARARGS | METH_KEYWORDS, ""},
    {"ztrsm", (PyCFunction)ztrsm, METH_VARARGS | METH_KEYWORDS, ""},
    {"chemm", (PyCFunction)chemm, METH_VARARGS | METH_KEYWORDS, ""},
    {"zhemm", (PyCFunction)zhemm, METH_VARARGS | METH_KEYWORDS, ""},
    {"cherk", (PyCFunction)cherk, METH_VARARGS | METH_KEYWORDS, ""},
    {"zherk", (PyCFunction)zherk, METH_VARARGS | METH_KEYWORDS, ""},
    {"cher2k", (PyCFunction)cher2k, METH_VARARGS | METH_KEYWORDS, ""},
    {"zher2k", (PyCFunction)zher2k, METH_VARARGS | METH_KEYWORDS, ""},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

const char *get_cblas_docs(void) {
  const char docs = 'w';
  const char *spam_doc = &docs;
  return spam_doc;
}

static struct PyModuleDef cblas_module = {
    PyModuleDef_HEAD_INIT,
    "pyblas_core",    /* __name__ of module */
    get_cblas_docs(), /* module documentation, may be NULL */
    -1,               /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    cblasmethods,
};

PyMODINIT_FUNC PyInit_pyblas_core(void) {
  PyObject *m;
  m = PyModule_Create(&cblas_module);
  if (m == NULL)
    return NULL;

  CallError = PyErr_NewException("call.error", NULL, NULL);
  if (PyModule_AddObject(m, "error", CallError) < 0) {
    Py_CLEAR(CallError);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
}
