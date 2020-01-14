from setuptools.command.build_ext import build_ext

import sys
import os
import glob
import setuptools

__version__ = '0.0.1'


# class get_pybind_include(object):
#     """Helper class to determine the pybind11 include path
#     The purpose of this class is to postpone importing pybind11
#     until it is actually installed, so that the ``get_include()``
#     method can be invoked. """
#
#     def __init__(self, user=False):
#         self.user = user
#
#     def __str__(self):
#         #print("CWD bind:", os.getcwd())
#         import pybind11
#         return pybind11.get_include(self.user)



# def get_sources(root, patterns):
#     #print("CWD :", os.getcwd())
#     sources = []
#     for p in patterns:
#         for path in glob.glob(os.path.join(root, p)):
#             print("Path: ", path)
#             sources.append(path)
#     return sources

# ext_modules = [
#     setuptools.Extension(
#         'bih',
#         get_sources('src', ['bih.cc', 'python_bih.cc']),
#         include_dirs=[
#             'src',
#             # Path to pybind11 headers
#             get_pybind_include(),
#             get_pybind_include(user=True)
#         ],
#         language='c++'
#     ),
# ]


# # As of Python 3.6, CCompiler has a `has_flag` method.
# # cf http://bugs.python.org/issue26689
# def has_flag(compiler, flagname):
#     """Return a boolean indicating whether a flag name is supported on
#     the specified compiler.
#     """
#     import tempfile
#     with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
#         f.write('int main (int argc, char **argv) { return 0; }')
#         try:
#             compiler.compile([f.name], extra_postargs=[flagname])
#         except setuptools.distutils.errors.CompileError:
#             return False
#     return True
#
#
# def cpp_flag(compiler):
#     """Return the -std=c++[11/14] compiler flag.
#     The c++14 is prefered over c++11 (when it is available).
#     """
#     if has_flag(compiler, '-std=c++14'):
#         return '-std=c++14'
#     elif has_flag(compiler, '-std=c++11'):
#         return '-std=c++11'
#     else:
#         raise RuntimeError('Unsupported compiler -- at least C++11 support '
#                            'is needed!')
#
#
# class BuildExt(build_ext):
#     """A custom build extension for adding compiler-specific options."""
#     c_opts = {
#         'msvc': ['/EHsc'],
#         'unix': [],
#     }
#
#     if sys.platform == 'darwin':
#         c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
#
#     def build_extensions(self):
#         ct = self.compiler.compiler_type
#         opts = self.c_opts.get(ct, [])
#         if ct == 'unix':
#             opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
#             opts.append(cpp_flag(self.compiler))
#             if has_flag(self.compiler, '-fvisibility=hidden'):
#                 opts.append('-fvisibility=hidden')
#         elif ct == 'msvc':
#             opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
#         for ext in self.extensions:
#             ext.extra_compile_args = opts
#         build_ext.build_extensions(self)
#

        
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='bgem',
    version=__version__,
    license='GPL 3.0',
    description='B-spline modelling CAD and meshing tools.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jan Brezina',
    author_email='jan.brezina@tul.cz',    
    url='https://github.com/geomop/pybs',

    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers        
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',        
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Scientific/Engineering',
    ],
    
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    
    packages=['bgem', 'bgem.polygons', 'bgem.bspline', 'bgem.gmsh', 'bgem.external'], #setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    #py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob('src/*.py')],
    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     #'': ['*.txt', '*.rst'],
    #     # And include any *.msg files found in the 'hello' package, too:
    #     #'hello': ['*.msg'],
    # },

    # include automatically all files in the template MANIFEST.in
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'bih', 'gmsh-sdk'],
    python_requires='>=3',
    # extras_require={
    #     # eg:
    #     #   'rst': ['docutils>=0.11'],
    #     #   ':python_version=="2.6"': ['argparse'],
    # },
    # entry_points={
    #     'console_scripts': [
    #         'nameless = nameless.cli:main',
    #     ]
    # },

    # ext_modules=ext_modules,
    # cmdclass={'build_ext': BuildExt}
    #test_suite='test.pytest_bih'
)        
        
        
        
