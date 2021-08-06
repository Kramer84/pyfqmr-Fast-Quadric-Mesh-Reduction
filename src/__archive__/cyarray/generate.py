import jinja2


pxds = []
pyxs = []

def fill_in_template(pytype, ctype, title, to_python_func, to_c_func,
                     implementation_preamble=None, definition_preamble=None,
                     buffer_type_code=None, sort_fn=None, sort_fn_reverse=None):
    if implementation_preamble is None:
        implementation_preamble = ''
    if definition_preamble is None:
        definition_preamble = ''
    if sort_fn and not sort_fn_reverse:
        implementation_preamble += '''

cdef int {sort_fn}_reverse(const void* a, const void* b) nogil:
    return -{sort_fn}(a, b)

'''.format(sort_fn=sort_fn)
        sort_fn_reverse = sort_fn + '_reverse'
    pxd_template = jinja2.Template(open("template.pxd", 'rt').read())
    print(title, sort_fn, sort_fn_reverse)
    with open("generated/{}_vector.pxd".format(ctype), 'wt') as fh:
        fh.write(pxd_template.render(
            pytype=pytype, ctype=ctype, to_python_func=to_python_func,
            to_c_func=to_c_func, title=title,
            definition_preamble=definition_preamble, sort_fn=sort_fn,
            sort_fn_reverse=sort_fn_reverse))
        pxds.append(fh.name)
    pyx_template = jinja2.Template(open("template.pyx", 'rt').read())
    with open("generated/{}_vector.pyx".format(ctype), 'wt') as fh:
        fh.write(pyx_template.render(
            pytype=pytype, ctype=ctype, to_python_func=to_python_func,
            to_c_func=to_c_func, title=title, implementation_preamble=implementation_preamble,
            buffer_type_code=buffer_type_code,
            sort_fn=sort_fn, sort_fn_reverse=sort_fn_reverse))
        pyxs.append(fh.name)


fill_in_template("object", "double", "Double", "PyFloat_FromDouble",
                 "PyFloat_AsDouble",
                 """from cpython.float cimport PyFloat_FromDouble, PyFloat_AsDouble
from libc.math cimport fabs

cdef int compare_value_double(const void* a, const void* b) nogil:
    cdef:
        double av, bv
    av = (<double*>a)[0]
    bv = (<double*>b)[0]
    if av < bv:
        return -1
    elif fabs(av - bv) < 1e-6:
        return 0
    else:
        return 1
""",
                 buffer_type_code="d", sort_fn="compare_value_double")
fill_in_template("object", "long", "Long", "PyInt_FromLong",
                 "PyInt_AsLong",
                 """from cpython.int cimport PyInt_FromLong, PyInt_AsLong

cdef int compare_value_long(const void* a, const void* b) nogil:
    cdef:
        long av, bv
    av = (<long*>a)[0]
    bv = (<long*>b)[0]
    if av < bv:
        return -1
    elif av == bv:
        return 0
    else:
        return 1
""",
                buffer_type_code="l", sort_fn="compare_value_long")

fill_in_template("object", "Py_ssize_t", "size_t", "PyInt_FromSsize_t",
                 "PyInt_AsSsize_t",
                 """from cpython.int cimport PyInt_FromSsize_t, PyInt_AsSsize_t

cdef int compare_value_Py_ssize_t(const void* a, const void* b) nogil:
    cdef:
        Py_ssize_t av, bv
    av = (<Py_ssize_t*>a)[0]
    bv = (<Py_ssize_t*>b)[0]
    if av < bv:
        return -1
    elif av == bv:
        return 0
    else:
        return 1
""",
                 buffer_type_code="pst", sort_fn="compare_value_Py_ssize_t")


with open("cyarray.pxd", 'wt') as fh:
    fh.write("""
cdef enum VectorStateEnum:
    should_free = 1

""")
    for pxd in pxds:
        fh.write("include \"%s\"\n" % pxd)
with open("cyarray.pyx", 'wt') as fh:
    for pyx in pyxs:
        fh.write("include \"%s\"\n" % pyx)
