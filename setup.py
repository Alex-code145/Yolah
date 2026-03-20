from setuptools import setup, Extension
from Cython.Build import cythonize
import sys

extensions = [
    Extension("Jeu._yolah_core", ["Jeu/_yolah_core.pyx"]),
]

setup(
    name="yolah_core",
    ext_modules=cythonize(extensions, annotate=False),
)
