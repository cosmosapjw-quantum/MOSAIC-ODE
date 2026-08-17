#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "weaveode/c_api.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

PyArrayObject *require_array(PyObject *object, int ndim, const char *name) {
    if (!PyArray_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return nullptr;
    }
    auto *array = reinterpret_cast<PyArrayObject *>(object);
    if (PyArray_TYPE(array) != NPY_DOUBLE) {
        PyErr_Format(PyExc_TypeError, "%s must have dtype float64", name);
        return nullptr;
    }
    if (PyArray_NDIM(array) != ndim) {
        PyErr_Format(PyExc_ValueError, "%s must have %d dimensions", name, ndim);
        return nullptr;
    }
    if (!PyArray_IS_C_CONTIGUOUS(array) || !PyArray_ISALIGNED(array)) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous and aligned", name);
        return nullptr;
    }
    return array;
}

bool raise_status(weaveode_status status, const char *operation) {
    if (status == WEAVEODE_OK) {
        return true;
    }
    PyObject *type = status == WEAVEODE_INVALID_ARGUMENT ? PyExc_ValueError : PyExc_RuntimeError;
    PyErr_Format(type, "%s failed with native status %d", operation, static_cast<int>(status));
    return false;
}

PyObject *py_wrms_scores(PyObject *, PyObject *args) {
    PyObject *residual_object = nullptr;
    PyObject *scale_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO:wrms_scores", &residual_object, &scale_object)) {
        return nullptr;
    }
    PyArrayObject *residuals = require_array(residual_object, 2, "residuals");
    PyArrayObject *scales = require_array(scale_object, 1, "scales");
    if (residuals == nullptr || scales == nullptr) {
        return nullptr;
    }
    const npy_intp n_candidates = PyArray_DIM(residuals, 0);
    const npy_intp dimension = PyArray_DIM(residuals, 1);
    if (PyArray_DIM(scales, 0) != dimension) {
        PyErr_SetString(PyExc_ValueError, "scales length must equal residual dimension");
        return nullptr;
    }
    npy_intp output_shape[1] = {n_candidates};
    PyObject *output_object = PyArray_SimpleNew(1, output_shape, NPY_DOUBLE);
    if (output_object == nullptr) {
        return nullptr;
    }
    auto *output = reinterpret_cast<PyArrayObject *>(output_object);
    const auto status = weaveode_wrms_scores(
        static_cast<const double *>(PyArray_DATA(residuals)),
        static_cast<std::size_t>(n_candidates),
        static_cast<std::size_t>(dimension),
        static_cast<const double *>(PyArray_DATA(scales)),
        static_cast<double *>(PyArray_DATA(output)));
    if (!raise_status(status, "wrms_scores")) {
        Py_DECREF(output_object);
        return nullptr;
    }
    return output_object;
}

PyObject *py_poly_bdf1_scores(PyObject *, PyObject *args) {
    PyObject *candidate_object = nullptr;
    PyObject *coefficient_object = nullptr;
    double y_prev = 0.0;
    double h = 0.0;
    double atol = 0.0;
    double rtol = 0.0;
    if (!PyArg_ParseTuple(args, "OddOdd:poly_bdf1_scores", &candidate_object, &y_prev, &h,
                          &coefficient_object, &atol, &rtol)) {
        return nullptr;
    }
    PyArrayObject *candidates = require_array(candidate_object, 1, "candidates");
    PyArrayObject *coefficients = require_array(coefficient_object, 1, "coefficients");
    if (candidates == nullptr || coefficients == nullptr) {
        return nullptr;
    }
    const npy_intp count = PyArray_DIM(candidates, 0);
    const npy_intp coefficient_count = PyArray_DIM(coefficients, 0);
    npy_intp output_shape[1] = {count};
    PyObject *output_object = PyArray_SimpleNew(1, output_shape, NPY_DOUBLE);
    if (output_object == nullptr) {
        return nullptr;
    }
    auto *output = reinterpret_cast<PyArrayObject *>(output_object);
    const auto status = weaveode_poly_bdf1_scores(
        static_cast<const double *>(PyArray_DATA(candidates)), static_cast<std::size_t>(count),
        y_prev, h, static_cast<const double *>(PyArray_DATA(coefficients)),
        static_cast<std::size_t>(coefficient_count), atol, rtol,
        static_cast<double *>(PyArray_DATA(output)));
    if (!raise_status(status, "poly_bdf1_scores")) {
        Py_DECREF(output_object);
        return nullptr;
    }
    return output_object;
}


PyObject *py_vector_bdf1_scores_from_rhs(PyObject *, PyObject *args) {
    PyObject *candidate_object = nullptr;
    PyObject *rhs_object = nullptr;
    PyObject *previous_object = nullptr;
    PyObject *atol_object = nullptr;
    double h = 0.0;
    double rtol = 0.0;
    if (!PyArg_ParseTuple(args, "OOOdOd:vector_bdf1_scores_from_rhs", &candidate_object,
                          &rhs_object, &previous_object, &h, &atol_object, &rtol)) {
        return nullptr;
    }
    PyArrayObject *candidates = require_array(candidate_object, 2, "candidates");
    PyArrayObject *rhs_values = require_array(rhs_object, 2, "rhs_values");
    PyArrayObject *previous = require_array(previous_object, 1, "previous");
    PyArrayObject *atol = require_array(atol_object, 1, "atol");
    if (candidates == nullptr || rhs_values == nullptr || previous == nullptr || atol == nullptr) {
        return nullptr;
    }
    const npy_intp n_candidates = PyArray_DIM(candidates, 0);
    const npy_intp dimension = PyArray_DIM(candidates, 1);
    if (PyArray_DIM(rhs_values, 0) != n_candidates || PyArray_DIM(rhs_values, 1) != dimension ||
        PyArray_DIM(previous, 0) != dimension || PyArray_DIM(atol, 0) != dimension) {
        PyErr_SetString(PyExc_ValueError, "vector BDF1 inputs have incompatible shapes");
        return nullptr;
    }
    npy_intp output_shape[1] = {n_candidates};
    PyObject *output_object = PyArray_SimpleNew(1, output_shape, NPY_DOUBLE);
    if (output_object == nullptr) {
        return nullptr;
    }
    auto *output = reinterpret_cast<PyArrayObject *>(output_object);
    const auto status = weaveode_vector_bdf1_scores_from_rhs(
        static_cast<const double *>(PyArray_DATA(candidates)),
        static_cast<const double *>(PyArray_DATA(rhs_values)),
        static_cast<std::size_t>(n_candidates), static_cast<std::size_t>(dimension),
        static_cast<const double *>(PyArray_DATA(previous)), h,
        static_cast<const double *>(PyArray_DATA(atol)), rtol,
        static_cast<double *>(PyArray_DATA(output)));
    if (!raise_status(status, "vector_bdf1_scores_from_rhs")) {
        Py_DECREF(output_object);
        return nullptr;
    }
    return output_object;
}

PyObject *py_radius_components(PyObject *, PyObject *args) {
    PyObject *point_object = nullptr;
    double radius = 0.0;
    if (!PyArg_ParseTuple(args, "Od:radius_components", &point_object, &radius)) {
        return nullptr;
    }
    PyArrayObject *points = require_array(point_object, 2, "points");
    if (points == nullptr) {
        return nullptr;
    }
    const npy_intp n_points = PyArray_DIM(points, 0);
    const npy_intp dimension = PyArray_DIM(points, 1);
    npy_intp output_shape[1] = {n_points};
    PyObject *output_object = PyArray_SimpleNew(1, output_shape, NPY_INT64);
    if (output_object == nullptr) {
        return nullptr;
    }
    auto *output = reinterpret_cast<PyArrayObject *>(output_object);
    const auto status = weaveode_radius_components(
        static_cast<const double *>(PyArray_DATA(points)), static_cast<std::size_t>(n_points),
        static_cast<std::size_t>(dimension), radius,
        static_cast<std::int64_t *>(PyArray_DATA(output)));
    if (!raise_status(status, "radius_components")) {
        Py_DECREF(output_object);
        return nullptr;
    }
    return output_object;
}

PyMethodDef methods[] = {
    {"wrms_scores", py_wrms_scores, METH_VARARGS,
     "Compute one weighted RMS residual score per candidate."},
    {"poly_bdf1_scores", py_poly_bdf1_scores, METH_VARARGS,
     "Score scalar polynomial BDF1 candidates using the original residual."},
    {"vector_bdf1_scores_from_rhs", py_vector_bdf1_scores_from_rhs, METH_VARARGS,
     "Score vector BDF1 candidates from precomputed RHS values."},
    {"radius_components", py_radius_components, METH_VARARGS,
     "Return connected-component labels for a radius graph."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native WeaveODE numerical primitives.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    return PyModule_Create(&module);
}
