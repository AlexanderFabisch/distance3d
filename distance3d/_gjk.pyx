cimport numpy as np
import numpy as np


cdef extern from "gilbert.c":
    double dsbp(int* nvs, int* ris, int* rjs, double** y, double** dell, double* zsol, double* als, int *backup)


cpdef distance_subalgorithm(
        int n_simplex_points,
        np.ndarray[long, ndim=1] old_indices_polytope1,
        np.ndarray[long, ndim=1] old_indices_polytope2,
        np.ndarray[double, ndim=2] simplex,
        np.ndarray[double, ndim=2] dot_product_table,
        np.ndarray[double, ndim=1] search_direction,
        np.ndarray[double, ndim=1] barycentric_coordinates,
        int backup):
    cdef int n_simplex_points_c = n_simplex_points
    cdef np.ndarray[int, ndim=1, mode="c"] old_indices_polytope1_c = np.ascontiguousarray(old_indices_polytope1, dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode="c"] old_indices_polytope2_c = np.ascontiguousarray(old_indices_polytope2, dtype=np.int32)
    cdef np.ndarray[double, ndim=2, mode="c"] simplex_c = np.ascontiguousarray(simplex)
    cdef np.ndarray[double, ndim=2, mode="c"] dot_product_table_c = np.ascontiguousarray(dot_product_table)
    cdef np.ndarray[double, ndim=1, mode="c"] search_direction_c = np.ascontiguousarray(search_direction)
    cdef np.ndarray[double, ndim=1, mode="c"] barycentric_coordinates_c = np.ascontiguousarray(barycentric_coordinates)
    cdef int backup_c = backup
    dstsq = dsbp(
        &n_simplex_points_c,
        &old_indices_polytope1_c[0],
        &old_indices_polytope2_c[0],
        <double**> &simplex_c[0, 0],
        <double**> &dot_product_table_c[0, 0],
        &search_direction_c[0],
        &barycentric_coordinates_c[0],
        &backup_c
    )
    return dstsq, n_simplex_points_c, backup_c
