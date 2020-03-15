from setuptools.command.build_ext import build_ext

import sys
import os
import glob
import setuptools

__version__ = '0.1.1'


        
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
    url='https://github.com/geomop/bgem',
    download_url='https://github.com/geomop/bgem/archive/v{__version__}.tar.gz',
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers        
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',               
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Scientific/Engineering',
    ],
    
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    
    packages=['bgem', 'bgem.polygons', 'bgem.bspline', 'bgem.gmsh', 'bgem.external', 'bgem.geometry'], #setuptools.find_packages(where='src'),
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
    #install_requires=['numpy', 'scipy', 'bih', 'gmsh-sdk<=4.5.1'],
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
        
        
        
