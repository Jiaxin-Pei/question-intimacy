import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="question_intimacy",
    version="1.1",
    author="Jiaxin Pei",
    author_email="pedropei@umich.edu",
    description="This package is used to predict intimacy for questions",
    python_requires=">=3.6",
    include_package_data=True,
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown"
    #package_data = {'':['data']}
)