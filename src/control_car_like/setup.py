from setuptools import setup

package_name = 'control_car_like'
lib = 'control_car_like/lib'
yolo6_lib = 'control_car_like/yolov6'
utils_lib = 'control_car_like/utils'
tracker_lib = 'control_car_like/tracker'
core_lib_yolov6 = 'control_car_like/yolov6/core'
data_lib_yolov6 = 'control_car_like/yolov6/data'
layers_lib_yolov6 = 'control_car_like/yolov6/layers'
models_lib_yolov6 = 'control_car_like/yolov6/models'
solver_lib_yolov6 = 'control_car_like/yolov6/solver'
utils_lib_yolov6 = 'control_car_like/yolov6/utils'



setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,
    lib,
    yolo6_lib,
    utils_lib,
    tracker_lib,
    core_lib_yolov6,
    data_lib_yolov6,
    layers_lib_yolov6,
    models_lib_yolov6,
    models_lib_yolov6,
    solver_lib_yolov6,
    utils_lib_yolov6
    ],


    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hero',
    maintainer_email='hero@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start = control_car_like.control_start:main',
            'test = control_car_like.control_start:main',
        ],
    },
)
