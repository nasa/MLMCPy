import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MLMCPy",
    version="0.0.01",
    author="James Warner ",
    author_email="james.e.warner@nasa.gov",
    description="Multi-Level Monte Carlo with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NASA/MLMCPy",
    packages=["MLMCPy",
              "MLMCPy.input",
              "MLMCPy.mlmc",
              "MLMCPy.model"],
    package_dir={'MLMCPy': 'MLMCPy'},
    install_requires=['numpy', 'scipy==1.0.0'],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
