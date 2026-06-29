import ctypes as C
import os
import sys
from os.path import dirname, exists, join, normpath


def get_plat_specifier():
    import distutils

    try:
        plat_name = distutils.util.get_platform()
    except AttributeError:
        plat_name = distutils.sys.platform
    plat_specifier = ".%s-%s" % (plat_name, sys.version[0:3])
    if hasattr(sys, "gettotalrefcount"):
        plat_specifier += "-pydebug"
    return plat_specifier


def get_lib_fname_list(libname):
    if sys.platform.startswith("linux"):
        spec_list = [get_plat_specifier(), "-manylinux1_x86_64", ""]
    else:
        spec_list = [get_plat_specifier(), ""]

    prefix_list = ["lib" + libname]
    if sys.platform.startswith("win32"):
        prefix_list.append(libname)
        ext = ".dll"
    elif sys.platform.startswith("darwin"):
        ext = ".dylib"
    elif sys.platform.startswith("linux"):
        ext = ".so"
    else:
        raise Exception("Unknown operating system: %s" % sys.platform)

    libnames = [
        "".join((prefix, spec, ext)) for spec in spec_list for prefix in prefix_list
    ]
    return libnames


def get_lib_dpath_list(root_dir):
    return [
        root_dir,
        join(root_dir, "lib"),
        join(root_dir, "build"),
        join(root_dir, "build", "lib"),
    ]


def find_lib_fpath(libname, root_dir, recurse_down=True, verbose=False):
    lib_fname_list = get_lib_fname_list(libname)
    tried_fpaths = []

    class FoundLib(Exception):
        pass

    FINAL_LIB_FPATH = None
    try:
        for lib_fname in lib_fname_list:
            curr_dpath = root_dir
            while curr_dpath is not None:
                for lib_dpath in get_lib_dpath_list(curr_dpath):
                    lib_fpath = normpath(join(lib_dpath, lib_fname))
                    tried_fpaths.append(lib_fpath)
                    if exists(lib_fpath):
                        FINAL_LIB_FPATH = lib_fpath
                        raise FoundLib
                _new_dpath = dirname(curr_dpath)
                if _new_dpath == curr_dpath:
                    curr_dpath = None
                else:
                    curr_dpath = _new_dpath
            if not recurse_down:
                break
    except FoundLib:
        return FINAL_LIB_FPATH

    raise ImportError("Cannot FIND dynamic library")


def load_clib(libname, root_dir):
    lib_fpath = find_lib_fpath(libname, root_dir)

    try:
        clib = C.cdll[lib_fpath]

        def def_cfunc(return_type, func_name, arg_type_list):
            cfunc = getattr(clib, func_name)
            cfunc.restype = return_type
            cfunc.argtypes = arg_type_list

        clib.__LIB_FPATH__ = lib_fpath
        return clib, def_cfunc, lib_fpath
    except OSError as ex:
        print("[C!] Caught OSError:\n%s" % ex)
        errsuffix = "Is there a missing dependency?"
    except Exception as ex:
        print("[C!] Caught Exception:\n%s" % ex)
        errsuffix = "Was the library correctly compiled?"

    print("[C!] cwd=%r" % os.getcwd())
    print("[C!] load_clib(libname=%r root_dir=%r)" % (libname, root_dir))
    print("[C!] lib_fpath = %r" % lib_fpath)
    errmsg = "[C] Cannot LOAD %r dynamic library. " % (libname,) + errsuffix
    print(errmsg)
    raise ImportError(errmsg)
