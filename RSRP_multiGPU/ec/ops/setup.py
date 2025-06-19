from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys

output_file_list = []
class CUDAExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources=sources, *args, **kwargs)

class CppExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources=sources, *args, **kwargs)

class CUDACppExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources=sources, *args, **kwargs)


class BuildExtension(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if isinstance(ext, CUDAExtension):
            # output_path = os.path.abspath(self.get_ext_fullpath(ext.name)).replace(self.get_ext_filename(ext.name), '')
            output_path = 'build'
            output_file = os.path.join(output_path, ext.name.split('.')[-1] + '.cu.o')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            nvcc_flags = [
                '--threads=4',
                '-Xcompiler', '-Wall,-fPIC,-fvisibility=hidden',
                '-ldl',
                '--expt-relaxed-constexpr',
                '-O3',
                '-DNDEBUG',
                '-Xcompiler', '-O3',
                '--generate-code=arch=compute_70,code=[compute_70,sm_70]',
                '--generate-code=arch=compute_75,code=[compute_75,sm_75]',
                '--generate-code=arch=compute_80,code=[compute_80,sm_80]',
                '--generate-code=arch=compute_86,code=[compute_86,sm_86]',
                '-x', 'cu', '-c',
                ext.sources[0], 
                '-o', output_file,
                '-I/usr/local/cuda/include'
            ]

            nvcc_command = ['nvcc'] + nvcc_flags
            self.spawn(nvcc_command)
            ext.extra_objects = [output_file]
            output_file_list.append(output_file)
        elif isinstance(ext, CppExtension):
            # output_path = os.path.abspath(self.get_ext_fullpath(ext.name)).replace(self.get_ext_filename(ext.name), '')
            output_path = 'build'
            output_file = os.path.join(output_path, ext.name.split('.')[-1] + '.cpp.o')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            python_executable = sys.executable
            python_path = sys.prefix
            python_version_info = sys.version_info
            if python_version_info.major == 3 and python_version_info.minor == 10:
                print("Python Version is 3.10")
            else:
                sys.exit(1)
            pybind_include_path = os.path.join(python_path, f'lib/python{python_version_info.major}.{python_version_info.minor}/site-packages/pybind11/include')
            # python_executable = "/usr/bin/python"
            # pybind_include_path = "/usr/include"  # Specify your pybind11 include path
            cuda_include_dir = "/usr/local/cuda/include"
            
            command = f'{python_executable}3-config --cflags'
            result = subprocess.run(command, shell=
            True, stdout=subprocess.PIPE)
            cflags = result.stdout.decode('utf-8').strip().split()
            cpp_flags = [
                '-I' + cuda_include_dir,
                '-I' + pybind_include_path,
                '-O3', '-DNDEBUG', '-O3', '-fPIC', '-fvisibility=hidden', '-flto', '-fno-fat-lto-objects',
                '-c', ext.sources[0],
                '-o', output_file
            ]+cflags
            cpp_command = ['c++'] + cpp_flags
            self.spawn(cpp_command)
            ext.extra_objects = [output_file]
            print(output_file)
            print(ext.extra_objects)
            output_file_list.append(output_file)
        elif isinstance(ext, CUDACppExtension):
            # output_path = os.path.abspath(self.get_ext_fullpath(ext.name)).replace(self.get_ext_filename(ext.name), '')
            output_path = 'build'
            python_executable = sys.executable
            command = f'{python_executable}3-config --extension-suffix'
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
            extension_suffix = result.stdout.decode('utf-8').strip()
            output_file = os.path.join(output_path, f'gpu_ops{extension_suffix}')
            cpp_flags = [
                '-fPIC','-O3','-DNDEBUG','-O3','-fPIC','-shared',
                '-o', output_file,
                output_file_list[0],output_file_list[1],
                '-L/usr/local/cuda/lib64',
                '-lcudadevrt','-lcudart_static','-lrt','-lpthread','-ldl',
                '-I/usr/local/cuda/include'
            ]
            cpp_command = ['c++'] + cpp_flags
            self.spawn(cpp_command)
            strip_command = ['strip'] + [output_file]
            self.spawn(strip_command)
            ext.extra_objects = [output_file]    
            print(output_file)
            print(ext.extra_objects)
        else:
            super().build_extension(ext)

setup(
    name='ec_cu_ops',
    version='0.0.1',
    author="BrainMatrix",
    author_email="brainmatrix2024@gmail.com",
    description="cuda extension for ec",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='matrix_vector_mul_batched',
            sources=['gpu_ops/matrix_vector_mul_batched.cu']
        ),
        CppExtension(
            name='gpu_ops',
            sources=['gpu_ops/gpu_ops.cpp']
        ),
        CUDACppExtension(
            name = 'gpu_ops',
            sources=[''],
        ),
    
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
