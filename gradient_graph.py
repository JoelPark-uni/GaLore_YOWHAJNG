import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 설정
# ----------------------------------------------------
STABLERANK_SAVE_DIR = "./gradient_rank/stable_rank"
PLOT_SAVE_DIR = "./gradient_rank/plots3"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# 앞에서 사용했던 모델 id들
MODEL_IDS = [0, 4, 5, 6, 62] # 0, 1, 2, 3, 4, 5, 6, 43, 55, 56, 58, 62

# 실험했던 batch size들
BATCH_SIZES = [2, 4, 8, 16, 32, 64, 128]  # 1, 2, 256, 512, 1024

# 🔥 model_id → 사람이 읽기 쉬운 label
MODEL_LABELS = {
    0: "base",
    1: "B1-trained",
    2: "B2-trained",
    3: "B4-trained",
    4: "B8-trained",
    5: "B16-trained",
    6: "B32-trained",
    43: "B64-trained",
    55: "B128-trained",
    56: "B256-trained",
    58: "B512-trained",
    62: "B1024-trained",
}


# ----------------------------------------------------
# 유틸 함수
# ----------------------------------------------------
def load_stable_rank_json(model_id, batch_size):
    path = os.path.join(
        STABLERANK_SAVE_DIR,
        f"stable_rank_mrpc_{model_id}_batch{batch_size}.json"
    )
    if not os.path.exists(path):
        print(f"[WARN] file not found: {path}")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data


def compute_mean_stable_rank(data, param_names=None):
    """
    param_names:
      - None이면: 전체 파라미터 평균
      - 리스트이면: 그 리스트에 들어있는 파라미터들만 평균
    """
    selected_means = []

    for name, ranks in data.items():
        if param_names is not None and name not in param_names:
            continue
        if len(ranks) == 0:
            continue
        selected_means.append(np.mean(ranks))

    if len(selected_means) == 0:
        return None

    return float(np.mean(selected_means))


def slugify(s: str) -> str:
    return (
        s.replace("/", "_")
         .replace(".", "_")
         .replace(" ", "_")
         .replace("[", "")
         .replace("]", "")
    )


# ----------------------------------------------------
# 파라미터 이름 77개 로드 (한 번만)
# ----------------------------------------------------
def get_param_names():
    for mid in MODEL_IDS:
        for bs in BATCH_SIZES:
            data = load_stable_rank_json(mid, bs)
            if data is not None and len(data) > 0:
                print(f"[INFO] Using param names from model_id={mid}, batch_size={bs}")
                return sorted(list(data.keys()))
    raise RuntimeError("No stable_rank json found to infer parameter names.")


# ----------------------------------------------------
# 레이어 이름 추출 & 레이어 그룹 만들기
# ----------------------------------------------------
def extract_layer_name(param_name: str):
    """
    예시 포맷을 가정:
      encoder.layer.0.attention.self.query.weight
      -> encoder.layer.0 를 레이어 이름으로 사용

    포맷이 다르면 이 함수만 수정하면 됨.
    """
    parts = param_name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] == "layer" and parts[i + 1].isdigit():
            # encoder.layer.0 까지 자르기
            return ".".join(parts[: i + 2])
    return None


def make_layer_groups(param_names):
    """
    layer_name -> [param_name1, param_name2, ...]
    """
    layer_groups = {}
    for name in param_names:
        layer_name = extract_layer_name(name)
        if layer_name is None:
            continue
        layer_groups.setdefault(layer_name, []).append(name)
    return layer_groups


# ----------------------------------------------------
# 메인: 전체 / 파라미터별 / 레이어별 그래프 생성
# ----------------------------------------------------
def main():
    param_names = get_param_names()
    layer_groups = make_layer_groups(param_names)

    # filters: (type, key)
    # type: "all" | "param" | "layer"
    filters = []

    # 1) 전체 평균
    filters.append(("all", None))

    # 2) 파라미터별 평균 (기존 동작)
    # for pn in param_names:
    #     filters.append(("param", pn))

    # 3) 레이어별 평균 (추가)
    for layer_name in sorted(layer_groups.keys()):
        filters.append(("layer", layer_name))

    for ftype, key in filters:
        if ftype == "all":
            title_suffix = "all_params"
            y_label = "Mean stable rank (avg over all params)"
            filename_suffix = "all_params"
            target_param_names = None  # 전체
        elif ftype == "param":
            title_suffix = key
            y_label = f"Mean stable rank ({key})"
            filename_suffix = slugify(key)
            target_param_names = [key]
        elif ftype == "layer":
            title_suffix = f"layer_avg: {key}"
            y_label = f"Mean stable rank (layer: {key})"
            filename_suffix = "layer_" + slugify(key)
            target_param_names = layer_groups[key]
        else:
            continue

        print(f"\n[DRAW] filter = {title_suffix}")

        plt.figure(figsize=(8, 6))

        for model_id in MODEL_IDS:
            xs, ys = [], []

            for bs in BATCH_SIZES:
                data = load_stable_rank_json(model_id, bs)
                if data is None:
                    continue

                m = compute_mean_stable_rank(data, param_names=target_param_names)
                if m is None:
                    continue

                xs.append(bs)
                ys.append(m)

            if len(xs) == 0:
                print(
                    f"[INFO] No usable data for model_id={model_id} "
                    f"with filter={title_suffix}"
                )
                continue

            # 사람이 읽기 쉬운 이름
            label = MODEL_LABELS.get(model_id, f"model_{model_id}")
            plt.plot(xs, ys, marker="o", label=label)

        plt.xscale("log", base=2)
        plt.xlabel("Batch size (log2 scale)")
        plt.ylabel(y_label)
        plt.title(f"Batch size vs Mean stable rank (MRPC)\n{title_suffix}")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(
            PLOT_SAVE_DIR,
            f"batch_vs_stablerank_{filename_suffix}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"  -> saved to {save_path}")


if __name__ == "__main__":
    main()
