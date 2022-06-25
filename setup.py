from setuptools import setup, find_packages

setup(
        name="mirt-iwvae",
        packages=find_packages("src"),
        package_dir={"": "src"})