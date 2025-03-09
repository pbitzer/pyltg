# Go get the version
from pathlib import Path

def get_version(relative_path):

    path_to_version = Path(__file__).resolve().parent.joinpath(relative_path)

    with open(path_to_version, 'r') as _file:
        for line in _file:
            print(line)
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]

with open("README.md", "r") as _file:
    long_descript = _file.read()

setuptools.setup(
    name='pyltg',
    version=get_version("pyltg/__init__.py"),
    author='Phillip Bitzer',
    author_email='bitzerp@uah.edu',
    description='A package for working with (mostly) lightning Data',
    long_description=long_descript,
    long_description_content_type='text/markdown',
    url='https://www.nsstc.uah.edu/users/phillip.bitzer/python_doc/pyltg/',
    keywords='lightning',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD-4-Clause',
        'Operating System :: OS Independent',
        ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'netcdf4',
        'pandas',
        'matplotlib',
        'scipy',
        'pyproj',
        'cartopy',
        'lmfit',
        'pip',
    ],
    extras_require={
        "LMA Flash Sort Read": ["pytables"],
    },
)