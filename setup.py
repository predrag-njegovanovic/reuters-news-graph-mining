from setuptools import setup
from setuptools import find_packages

NAME = "reuters-news-graph-mining"
AUTHOR = "Predrag Njegovanovic"
VERSION = "1.0"
EMAIL = "djaps94@gmail.com"
DESCRIPTION = "Analysis of Retuers News graph properties"
MIT = "MIT"
URL = "https://github.com/Djaps94/reuters-news-graph-mining"

setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        license=MIT,
        url=URL,
        packages=find_packages() + ['data', 'figures', 'results'],
        include_package_data=True
    )
