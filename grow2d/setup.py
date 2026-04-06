from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [Extension(
            "grow2d",
            sources=["src/grow2d.pyx"],
            include_dirs=["src"],
            define_macros=[("GROW2D_IMPLEMENTATION", None)],
            extra_compile_args=["-O3"],
            language="c",
        )],
        compiler_directives={"language_level": "3"},
    )
)
