"""
Setup script for binarycpython
"""

import distutils.command.build
import os
import re
import subprocess
import sys
from distutils.core import Extension, setup

#
this_file = os.path.abspath(__file__)
this_file_dir = os.path.dirname(this_file)

###
REQUIRED_BINARY_C_VERSIONS = ["2.2.4"]

############################################################
# Defining functionality
############################################################


# Functions
def version():
    """
    opens VERSION and returns version number
    """

    with open("VERSION") as file:
        return file.read().strip()


def readme():
    """
    Opens readme file and returns content
    """

    with open("README.md") as file:
        return file.read()


def license():
    """
    Opens license file and returns the content
    """

    with open("LICENSE.md") as file:
        return file.read()


def requirements(directory):
    """
    Opens requirements.txt and returns content as a list
    """

    requirements_file = os.path.join(directory, "requirements.txt")

    # Read out file and construct list
    requirements_list = []
    with open(requirements_file) as f:
        for el in f.readlines():
            requirements_list.append(el.strip())

    return requirements_list


def check_version(installed_binary_c_version, required_binary_c_versions):
    """Function to check the installed version and compare it to the required version"""
    message = """
    Something went wrong. Make sure that binary_c config exists.
    Possibly the binary_c version that is installed ({}) does not match the binary_c versions ({})
    that this release of the binary_c python module requires.
    """.format(
        installed_binary_c_version, required_binary_c_versions
    )
    assert installed_binary_c_version in required_binary_c_versions, message


def execute_make():
    """
    Function to execute the makefile.

    This makefile builds the binary_c_python_api library that python will use to interface wth
    """

    # Custom extra command:
    make_command = ["make"]

    p = subprocess.run(make_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = p.stdout  # stdout = normal output
    stderr = p.stderr  # stderr = error output

    if p.returncode != 0:
        print("Something went wrong when executing the makefile:")
        print(stderr.decode("utf-8"))
        print("Aborting")
        sys.exit(-1)

    else:
        print(stdout.decode("utf-8"))
        print("Successfully built the libbinary_c_api.so")


def call_binary_c_config(binary_c_dir, command):
    """
    Function to call the binary_c config
    """

    binary_c_config = os.path.join(BINARY_C_DIR, "binary_c-config")

    command_result = (
        subprocess.run([binary_c_config, command], stdout=subprocess.PIPE, check=True)
        .stdout.decode("utf-8")
        .split()
    )

    return command_result


############################################################
# First level checks
############################################################

####
GSL_DIR = os.getenv("GSL_DIR", None)
if not GSL_DIR:
    print(
        "Warning: GSL_DIR is not set, this might lead to errors along the installation if\
        there is no other version of GSL in the include dirs"
    )
BINARY_C_DIR = os.getenv("BINARY_C", None)
if not BINARY_C_DIR:
    print(
        "\n\n****\n**** Error: the BINARY_C environment variable is not set.\n**** This environment variable should point to the root of your binary_c\n**** installation (i.e. the directory you acquired from the repository).\n**** Aborting setup.\n****\n\n"
    )
    quit(1)

############################################################
# Getting information from binary_c
############################################################

# binary_c must be installed.
BINARY_C_VERSION = call_binary_c_config(BINARY_C_DIR, "version")
check_version(BINARY_C_VERSION[0], REQUIRED_BINARY_C_VERSIONS)

BINARY_C_INCDIRS = call_binary_c_config(BINARY_C_DIR, "incdirs_list")
BINARY_C_LIBDIRS = call_binary_c_config(BINARY_C_DIR, "libdirs_list")
binary_c_cflags = call_binary_c_config(BINARY_C_DIR, "cflags")
BINARY_C_CFLAGS = binary_c_cflags
# BINARY_C_CFLAGS.remove('-fvisibility=hidden')
BINARY_C_LIBS = call_binary_c_config(BINARY_C_DIR, "libs_list")
BINARY_C_DEFINES = call_binary_c_config(BINARY_C_DIR, "define_macros")

# create list of tuples of defined macros
BINARY_C_DEFINE_MACROS = []
LONE = re.compile("^-D(.+)$")
PARTNER = re.compile("^-D(.+)=(.+)$")
for x in BINARY_C_DEFINES:
    y = PARTNER.match(x)
    if y:
        BINARY_C_DEFINE_MACROS.extend([(y.group(1), y.group(2))])
    else:
        y = LONE.match(x)
        if y:
            BINARY_C_DEFINE_MACROS.extend([(y.group(1), None)])

# create a list of defined features/present libraries
BINARY_C_ATTRIBUTE_MACROS = []
for flag in binary_c_cflags:
    if flag.startswith("-D__HAVE_"):
        y = LONE.match(flag)
        if y:
            BINARY_C_ATTRIBUTE_MACROS.extend([(y.group(1), "1")])

# add API header file
API_h = os.path.join(BINARY_C_DIR, "src", "API", "binary_c_API.h")

############################################################
# Determine all directories and libraries
############################################################

INCLUDE_DIRS = [
    os.path.join(BINARY_C_DIR, "src"),
    os.path.join(BINARY_C_DIR, "src", "API"),
    "include",
] + BINARY_C_INCDIRS
if GSL_DIR:
    INCLUDE_DIRS += [os.path.join(GSL_DIR, "include")]

# LIBRARIES = ["binary_c"] + BINARY_C_LIBS + ["binary_c_python_api"]
LIBRARIES = ["binary_c"] + BINARY_C_LIBS

LIBRARY_DIRS = [
    os.path.join(BINARY_C_DIR, "src"),
    "./",
    os.path.join(this_file_dir, "lib/"),
    os.path.join(this_file_dir, "binarycpython/"),
] + BINARY_C_LIBDIRS

RUNTIME_LIBRARY_DIRS = [
    os.path.join(BINARY_C_DIR, "src"),
    "./",
    os.path.join(this_file_dir, "lib/"),
] + BINARY_C_LIBDIRS

# filter out duplicates
INCLUDE_DIRS = list(dict.fromkeys(INCLUDE_DIRS))
BINARY_C_LIBS = list(dict.fromkeys(BINARY_C_LIBS))
LIBRARIES = list(dict.fromkeys(LIBRARIES))
LIBRARY_DIRS = list(dict.fromkeys(LIBRARY_DIRS))
RUNTIME_LIBRARY_DIRS = list(dict.fromkeys(RUNTIME_LIBRARY_DIRS))

############################################################
# Making the python-c binding module
############################################################

BINARY_C_PYTHON_API_MODULE = Extension(
    name="binarycpython._binary_c_bindings",
    sources=["src/binary_c_python.c"],
    include_dirs=INCLUDE_DIRS,
    libraries=LIBRARIES,
    library_dirs=LIBRARY_DIRS,
    runtime_library_dirs=RUNTIME_LIBRARY_DIRS,
    define_macros=[] + BINARY_C_DEFINE_MACROS + BINARY_C_ATTRIBUTE_MACROS,
    extra_objects=[],
    extra_compile_args=[],
    language="C",
)
headers = ["src/includes/header.h"]

############################################################
# Custom build command
############################################################


# Override build command
class CustomBuildCommand(distutils.command.build.build):
    def run(self):
        # Run the original build command
        distutils.command.build.build.run(self)


############################################################
# Main setup function call
############################################################

setup(
    name="binarycpython",
    version=version(),
    description="""This is a python API for binary_c (versions {}) by David Hendriks, Rob Izzard and collaborators. Based on the initial set up by Jeff andrews.""".format(
        ",".join(REQUIRED_BINARY_C_VERSIONS),
    ),
    author="David Hendriks",
    author_email="davidhendriks93@gmail.com",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://binary_c.gitlab.io/binary_c-python",
    license="gpl",
    keywords=[
        "binary_c",
        "astrophysics",
        "stellar evolution",
        "population synthesis",
    ],  # Keywords that define your package best
    packages=[
        "binarycpython",
        "binarycpython.core",
        "binarycpython.utils",
        "binarycpython.utils.population_extensions",
        "binarycpython.utils.convolution",
        "binarycpython.tests",
        "binarycpython.tests.tests_population_extensions",
    ],
    package_data={
        "": [
            "example_data/*.dat",
            "example_data/*.json",
        ],
        # "convolution": [
        #     "example_data",
        #     "example_data",
        # ],
    },
    install_requires=requirements(this_file_dir),
    include_package_data=True,
    ext_modules=[BINARY_C_PYTHON_API_MODULE],  # binary_c must be loaded
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={"build": CustomBuildCommand},
)
