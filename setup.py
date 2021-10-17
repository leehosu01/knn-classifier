import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

import glob
def recursive_dir_walker(dir):
    files = [recursive_dir_walker(directory) for directory in glob.glob(f'{dir}/*')]
    if len(files) == 0: files = [[dir]]
    return sum(files, [])

datafiles = []#recursive_dir_walker("ABC/*")

setuptools.setup(
    name="knn classifier",
    version="0.0.1",
    author="Hosu Lee",
    author_email="leehosu01@naver.com",
    description="cosine similarity weighted classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leehosu01/knn-classifier",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license_files = ('LICENSE.txt',),
    python_requires='>=3.6',
    install_requires=requirements,
    data_files = datafiles,
    include_package_data=True
)