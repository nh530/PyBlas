from skbuild import setup

# NOTE: Very tricky; package_dir and cmake_install_dir have non-trivial, non-obvious effects on package installation.
setup(
    name="pyblas",
    version="0.5.0",
    description="Multi-threaded matrix mathematics",
    author='Norman Hong',
    license="MIT",
    packages=['pyblas'],
    package_dir={'pyblas': 'src/pyblas'},
    python_requires=">=3.9",
)
