# Pointcept PTv3 en RTX 5090

Este paquete crea una imagen con los componentes críticos fijados para una RTX
5090 (`sm_120`) con:

- Ubuntu 24.04 y toolkit CUDA 12.8;
- PyTorch 2.11.0 `cu128`;
- Pointcept 1.7.0 fijado a un commit concreto;
- `spconv`/`cumm` compilables para Blackwell mediante forks fijados;
- la API de FlashAttention que usa PTv3, implementada sobre la atención
  fusionada SDPA/cuDNN de PyTorch.

## Por qué no instala `flash-attn` de Dao directamente

La RTX 5090 tiene compute capability 12.0. FlashAttention-2 no declara soporte
oficial para Blackwell de consumo y existen fallos abiertos de compilación y
ejecución en `sm_120`. Forzar `TORCH_CUDA_ARCH_LIST=...+PTX` no convierte ese
código en compatible.

PTv3 solo importa `flash_attn_varlen_qkvpacked_func`. La pequeña capa
`flash_attn_compat` proporciona exactamente esa llamada usando SDPA fusionado.
Excluye expresamente el backend matemático de memoria cuadrática: si cuDNN no
puede ejecutar una forma concreta, el programa falla con un error claro en vez
de caer silenciosamente al camino que provoca OOM. Debes mantener AMP activo.

No ejecutes `pip install flash-attn` dentro de esta imagen: sustituiría la capa
compatible por una extensión que actualmente no es fiable para `sm_120`.

## Requisito del host

Necesitas Docker y NVIDIA Container Toolkit. El valor `CUDA Version: 13.1` de
`nvidia-smi` indica la versión máxima de CUDA admitida por el driver; no obliga
a que el contenedor use CUDA 13.1. El driver 591.86 puede ejecutar esta imagen
CUDA 12.8.

Comprueba primero el paso de la GPU al contenedor:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Construcción y verificación

Desde esta carpeta:

```bash
docker compose build --no-cache
docker compose run --rm ptv3 verify --iterations 5
```

El primer uso de `spconv` puede tardar varios minutos porque compila kernels
JIT para `sm_120`. La comprobación correcta termina exactamente con:

```text
RTX5090_PTV3_ALL_CHECKS_OK
```

La prueba ejecuta, en la RTX 5090 real:

1. CUDA 12.8 y compute capability 12.0;
2. atención fusionada con forward y backward;
3. `spconv.SubMConv3d` repetido con forward y backward;
4. un Pointcept PTv3 pequeño, con Flash Attention y AMP, también repetido.

No consideres validada la instalación si no aparece el marcador final.

## Usar tu proyecto

Abre una terminal con la copia de Pointcept incluida en la imagen:

```bash
docker compose run --rm ptv3
```

Para montar tu repositorio en `/workspace`, usa una ruta absoluta:

```bash
POINTCEPT_WORKSPACE=/ruta/absoluta/a/tu/proyecto \
  docker compose run --rm ptv3 verify --iterations 5

POINTCEPT_WORKSPACE=/ruta/absoluta/a/tu/proyecto \
  docker compose run --rm ptv3 python main.py
```

También puedes prescindir de Compose:

```bash
docker build --no-cache -t pointcept-ptv3-rtx5090:cu128 .

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v /ruta/absoluta/a/tu/proyecto:/workspace \
  pointcept-ptv3-rtx5090:cu128 verify --iterations 5
```

En la configuración de PTv3 conserva:

```python
enable_flash=True
enable_rpe=False
upcast_attention=False
upcast_softmax=False
```

y entrena con AMP (`float16` o `bfloat16`). Son las restricciones del camino
Flash de PTv3.

## PointOps

PTv3 no necesita PointOps. Si tu proyecto también ejecuta PTv1, PTv2 o una
evaluación que sí lo importe, compílalo después de montar el repositorio:

```bash
install-pointops /workspace/libs/pointops
```

## Límites reales de la solución

La atención deja de ser el problema: se obliga a usar un kernel fusionado de
memoria eficiente. El componente todavía experimental en una RTX 5090 es
`spconv`. No hay wheel oficial CUDA 12.8/`sm_120`; esta imagen usa revisiones
fijadas de forks con soporte Blackwell. La prueba incluida detecta los fallos
más comunes, pero ningún Docker puede prometer que un entrenamiento largo y
un dataset desconocido no expongan un kernel o una forma aún no soportados.

Si el verificador falla, conserva la salida completa. En particular, un error
`IndexKernel` o `ScatterGatherKernel` después de que la atención haya pasado
apunta a `spconv`, no a FlashAttention ni a falta de VRAM.

## Versiones fijadas

| Componente | Versión/revisión |
|---|---|
| Imagen CUDA | `12.8.1-cudnn-devel-ubuntu24.04` |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| torchaudio | `2.11.0+cu128` |
| Pointcept | `81ab8f5c4781bf63ed76d6e97e2d7241e4ceffc8` |
| cumm fork | `97aa43a4fe37f46b94f54cf96a7177a945c2e0d4` |
| spconv fork | `c160d84ac62ff4653b8a73f80630039fcabc92ac` |

## Referencias

- [Pointcept](https://github.com/Pointcept/Pointcept)
- [Versiones CUDA oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/)
- [Compatibilidad oficial de FlashAttention](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)
- [Fallo de FlashAttention-2 en `sm_120`](https://github.com/Dao-AILab/flash-attention/issues/2361)
- [Estado de FlashAttention-4 en RTX 50](https://github.com/Dao-AILab/flash-attention/issues/2453)
- [Seguimiento de `spconv` en RTX 5090](https://github.com/traveller59/spconv/issues/746)
- [Seguimiento de PTv3 en RTX 5090](https://github.com/Pointcept/PointTransformerV3/issues/159)
- [Receta comunitaria `spconv` CUDA 12.8/Blackwell](https://github.com/davidzha712/spconv-blackwell-cu128)
