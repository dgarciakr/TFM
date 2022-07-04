Esta carpeta contiene los entornos desarrollados durante el trabajo de fin de máster

## Instalación
1. Instalar [OpenAi Gym](https://github.com/openai/gym)
```bash
pip install gym
```

2. Descargar e instalar `gym-hpc`
```bash
git clone https://github.com/hpc-unex/gym-hpc
cd gym-hpc
pip install -e .
```
## Uso de los entornos

Una vez instalados los entornos no hay más que cargarlos al modelo con: 'gym_hpc:Mapping-v0' o 'gym_hpc:Mapping-v1'

##Librerías:
- Gym = 0.19.0
- Mpi = 1.0.0
- Numpy = 1.19.5
- Setuptools = 45.2.0
- Torch = 1.10.1
