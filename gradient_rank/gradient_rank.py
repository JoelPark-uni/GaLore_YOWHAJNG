import os
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json

# ==========================
# 공통 설정
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# 실험할 batch size들
BATCH_SIZES = [256, 2, 1] # 1024, 512 # 128, 64, 32, 16, 8, 4

# stable rank 저장 위치
STABLERANK_SAVE_DIR = "./gradient_rank/stable_rank"
os.makedirs(STABLERANK_SAVE_DIR, exist_ok=True)

# 실험할 모델들: (모델 경로(or 이름), id 숫자)
MODEL_SPECS = [
    ("roberta-base", 0),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__1", 1),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__2", 2),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__3", 3),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__4", 4),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__5", 5),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__6", 6),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__43", 43),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__55", 55),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__56", 56),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__58", 58),
    ("../results/ft/roberta_base_mrpc_T500/mrpc__62", 62),
]


# ==========================
# stable rank util
# stable_rank(A) = ||A||_F^2 / ||A||_2^2
# ==========================
def tensor_to_matrix(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.view(-1, 1)
    return t.view(t.shape[0], -1)


def stable_rank(grad: torch.Tensor) -> float:
    mat = tensor_to_matrix(grad)
    fro = torch.linalg.norm(mat, ord="fro")
    spec = torch.linalg.norm(mat, ord=2)
    if spec.item() == 0.0:
        return 0.0
    return float((fro ** 2 / spec ** 2).item())


# ==========================
# MRPC 원본 데이터 로드 (텍스트 기준, 한 번만)
# ==========================
raw_datasets = load_dataset("glue", "mrpc")
train_dataset = raw_datasets["train"]
num_samples = len(train_dataset)
print(f"MRPC train samples: {num_samples}")


# ==========================
# 모델별 루프
# ==========================
for model_path, model_id in MODEL_SPECS:
    print("=" * 100)
    print(f"Processing model: {model_path} (id={model_id})")
    print("=" * 100)

    # 1) 모델 / 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()

    # 2) MRPC 인코딩 (모델별 tokenizer 사용)
    def preprocess_mrpc(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    encoded = train_dataset.map(preprocess_mrpc, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in encoded.column_names:
        cols.append("token_type_ids")
    encoded.set_format(type="torch", columns=cols)

    # 3) stable rank를 구할 파라미터 선택 (2D weight만)
    target_params = {
        name: p for name, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2
    }
    print("num target params:", len(target_params))

    # 4) batch size별로 바로 gradient → stable rank 계산
    for bs in BATCH_SIZES:
        print(f"\n[model_id={model_id}] Computing stable ranks with batch_size={bs}")
        dataloader = DataLoader(encoded, batch_size=bs, shuffle=False)

        # param_name -> [stable_rank(batch0), stable_rank(batch1), ...]
        per_param_ranks = {name: [] for name in target_params.keys()}

        for batch in tqdm(dataloader, desc=f"batch_size={bs}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # grad 초기화
            model.zero_grad(set_to_none=True)

            labels = batch["label"]
            inputs = {k: v for k, v in batch.items() if k != "label"}

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            # 이 batch에 대한 각 파라미터의 gradient로 stable rank 계산
            for name, p in target_params.items():
                if p.grad is None:
                    continue
                g = p.grad.detach().cpu()
                r = stable_rank(g)
                per_param_ranks[name].append(r)

        # 5) JSON 저장
        out_path = os.path.join(
            STABLERANK_SAVE_DIR,
            f"stable_rank_mrpc_{model_id}_batch{bs}.json"
        )
        # float로 캐스팅
        serializable = {
            name: [float(r) for r in ranks]
            for name, ranks in per_param_ranks.items()
        }
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)

        print(
            f"Saved stable ranks for model_id={model_id}, "
            f"batch_size={bs} to {out_path}"
        )

    # 6) 메모리 정리
    del model
    torch.cuda.empty_cache()
    print(f"\nDone for model: {model_path} (id={model_id})\n")
