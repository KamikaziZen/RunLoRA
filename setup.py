from setuptools import setup

setup(
   name='runlora',
   version='1.0.0',
   license='MIT',
   author='Aleksandr Mikhalev and Daria Cherniuk',
   author_email='al.mikhalev@skoltech.ru, daria.cherniuk@skoltech.ru, kamikazizen@gmail.com',
   description='Fast and Furious lora implementations',
   packages=['runlora'],
   install_requires=['transformers', 'peft', 'bitsandbytes', 'scipy==1.10.1']
)
