# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="icbs",
    version="0.1",
    author="Amazon Quantum Solutions Lab",
    author_email="gilir@amazon.com, johbruba@amazon.com",
    description="iCBS: Iterative Combinatorial Brain Surgeon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)
