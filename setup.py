"""
@File : setup.py
@Author : 杨与桦
@Time : 2023/10/29 9:21
"""
import setuptools  # 导入setuptools打包工具

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="traffictools-pack",  # 用自己的名替换其中的YOUR_USERNAME_
    version="0.0",  # 包版本号，便于维护版本
    author="Yang",  # 作者，可以写自己的姓名
    author_email="2803680027@qq.com",  # 作者联系方式，可写自己的邮箱地址
    description="A toolkit for undergraduate transportation students",  # 包的简述
    url="https://github.com/YangYuhua122121/traffictools/tree/main",  # 自己项目地址，比如github的项目地址
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # 对python的最低版本要求
)
