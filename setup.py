import sys
from setuptools import setup, find_packages

if sys.version_info[0] < 3:
  import imp
  VERSION = imp.load_source('rlhf.version', 'rlhf/utils/version.py').VERSION
else:
  from importlib.machinery import SourceFileLoader
  VERSION = SourceFileLoader("rlhf.version", "rlhf/utils/version.py") \
      .load_module().VERSION

setup(
    name='rlhf',
    version=VERSION,
    python_requires='>3.6.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    author='Aliyun Inc.',
    license='Apache 2.0',
)
