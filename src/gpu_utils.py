"""GPU の使用量を抑えるための共通ユーティリティ。

共有計算機（gpu-a03 など）で他のユーザーと GPU を分け合うために使う。
環境変数で制御するので、スクリプト側のコード変更なしに効かせられる。

    KERMUT_GPU_MEM_FRACTION   1プロセスが確保できる VRAM の上限（0〜1。既定 0.8）
    KERMUT_GPU_THREADS        CPU スレッド数の上限（省略時は変更しない）

CUDA_VISIBLE_DEVICES は PyTorch ではなくシェル側で指定する
（このモジュールが import される時点では既に手遅れなため）:

    CUDA_VISIBLE_DEVICES=2 python ...

注意: set_per_process_memory_fraction はソフトな上限で、超えると確保が
失敗して OOM エラーになる（他プロセスのメモリを守るための仕組み）。
「上限まで使い切ってから遅くなる」のではなく「上限を超えたら落ちる」挙動。
"""

import os

import torch


def limit_gpu_usage(verbose: bool = True) -> None:
    """環境変数に従って GPU / CPU の使用量に上限をかける。

    CUDA が使えない環境では何もしない（CPU 実行を妨げない）。
    """
    # CPU スレッド数。GPU を待つ間に CPU を占有しすぎるのを防ぐ
    n_threads = os.environ.get("KERMUT_GPU_THREADS")
    if n_threads:
        torch.set_num_threads(int(n_threads))
        if verbose:
            print(f"[gpu_utils] CPU スレッド数を {n_threads} に制限")

    if not torch.cuda.is_available():
        return

    frac = float(os.environ.get("KERMUT_GPU_MEM_FRACTION", "0.8"))
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"KERMUT_GPU_MEM_FRACTION は 0〜1 で指定する: {frac}")

    device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(frac, device)

    if verbose:
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        name = torch.cuda.get_device_name(device)
        # CUDA_VISIBLE_DEVICES で絞っている場合、ここでの device 番号は
        # 絞った後の連番なので、物理的な番号は環境変数の方を見せる
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(未指定=全GPU可視)")
        print(
            f"[gpu_utils] GPU={name} CUDA_VISIBLE_DEVICES={visible} "
            f"VRAM上限={frac:.0%} ({total * frac:.1f}GiB / {total:.1f}GiB)"
        )
