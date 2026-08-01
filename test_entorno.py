import torch
import time

print("="*50)
print("🔍 INICIANDO TEST DEL ENTORNO PTv3 PARA BLACKWELL")
print("="*50)

# 1. Comprobar PyTorch y Hardware
print("\n[1] Comprobando PyTorch y GPU...")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("❌ ERROR: No se detecta la GPU.")
    exit(1)

# 2. Comprobar PointOps
print("\n[2] Comprobando PointOps (C++ Custom)...")
try:
    import pointops
    print("✅ PointOps importado correctamente. (Compilado para Hopper+PTX/Blackwell)")
except Exception as e:
    print(f"❌ ERROR importando pointops: {e}")

# 3. Comprobar SpConv y Torch Geometric
print("\n[3] Comprobando dependencias Sparse (SpConv y PyG)...")
try:
    import spconv.pytorch as spconv
    import torch_geometric
    print(f"✅ Torch Geometric Version: {torch_geometric.__version__}")
    
    # Test rápido de SpConv
    features = torch.randn(1, 32).cuda()
    indices = torch.tensor([[0, 0, 0, 0]]).cuda().int()
    x = spconv.SparseConvTensor(features, indices, [16, 16, 16], 1)
    print("✅ SpConv Tensor creado correctamente en la GPU.")
except Exception as e:
    print(f"❌ ERROR con SpConv o PyG: {e}")

# 4. Comprobar la atención fusionada utilizada realmente por PTv3
print("\n[4] Comprobando atención fusionada PTv3...")

try:
    import flash_attn
    from flash_attn import flash_attn_varlen_qkvpacked_func

    print(f"✅ flash_attn compatible: {flash_attn.__version__}")

    # PTv3 trabaja con secuencias/patches empaquetados.
    batch_size = 4
    seqlen = 1024
    nheads = 8
    headdim = 64
    tokens = batch_size * seqlen

    print(
        f"-> Generando QKV empaquetado: "
        f"({tokens}, 3, {nheads}, {headdim})"
    )

    qkv = torch.randn(
        tokens,
        3,
        nheads,
        headdim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        seqlen,
        device="cuda",
        dtype=torch.int32,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.time()

    out = flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens,
        max_seqlen=seqlen,
        dropout_p=0.0,
    )

    # Verificar también el backward, imprescindible para entrenar.
    loss = out.float().square().mean()
    loss.backward()

    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000
    peak_mib = torch.cuda.max_memory_allocated() / 2**20

    assert out.shape == (tokens, nheads, headdim)
    assert qkv.grad is not None
    assert torch.isfinite(qkv.grad).all()

    print(f"✅ Forward/backward fusionado: {elapsed_ms:.2f} ms")
    print(f"-> Salida: {tuple(out.shape)}")
    print(f"-> Pico de memoria CUDA: {peak_mib:.1f} MiB")
    print("\n🎉 Atención eficiente de PTv3 funcionando correctamente.")

except Exception as e:
    print(f"\n❌ ERROR en la atención fusionada de PTv3: {e}")