# train > model > prepare > inference 순서로 봐야함 (일주일에 파일 하나는 봐야함)
# 한 줄 한 줄 의미를 완전히 이해할 수 있어야함 (모르는 부분은 chatGPT, 멘토님께 물어보기)
# 코드를 읽고 설명 자세하게 가능한 수준이 되어야 함
# ModernBERT로 바꿀 수 있어야 함 (포트폴리오)
# 옆에 두고 수정할 수 있어야 함

# 학습 실행 방법
# 1) single GPU에서 실행
# python train.py --batch_size=21 --compile=False

# 2) 1개의 노드에서 4개의 GPU를 사용 (DDP 실행)
# torchrun --standalone --nproc_per_node=4 train.py

# 3) 2개의 노드를 활용한 DDP 실행 (예시 IP: 123.456.123.456)
# (1) master 노드에서 실행
#  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# (2) worker 노드에서 실행
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py


import os
import time
import math
import pickle
import wandb
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP # 여러 GPU를 사용해서 분산학습 가능하게 하는 라이브러리
from torch.distributed import init_process_group, destroy_process_group # 분산학습을 시작하고 종료하는 함수
from model import GPTConfig, GPT # GPT 모델 관련 클래스 (model.py)에서 불러옴

# -----------------------------------------------------------------------------
# GPU 1개와 DDP 방식으로 GPT-2 Pretraining 사전학습하기

# 모델: GPT-2 (124M)
# 학습데이터: OpenWebText
# 학습방법: GPU 1개 & Distributed Data Parallel(DDP) 모드

# 기본 config 값

# 1 batch = 1 step
# 모델 저장 장소 및 저장 방식 설정
out_dir = 'out' # 모델 저장 디렉토리
eval_interval = 20 # 2000 step 마다 평가 실시
log_interval = 1 # 로그를 1 step 마다 출력
eval_iters = 20 # 평가 시 200 step 만 loss 측정
eval_only = False # True이면 평가 후 바로 종료
always_save_checkpoint = True # 매 평가 후 체크포인트 저장함

# 모델 초기화 (init_from) 모드: 다음 중 1개 선택
# 'scratch': 새로운 모델을 생성
# 'resume': 이전 학습된 체크포인트(ckpt.pt)에서 모델을 불러오기
# 'gpt2': 사전학습된 GPT-2 모델을 불러오기 # 꼭 해보기 ex) 어느 체크포인트에서 가져와서 돌려주세요
init_from = 'gpt2'

# Weights & Biases (W&B): 시각화를 위한 툴 (대시보드) > 많이 사용
wandb_log = True # False로 할 경우 weights와 bias 로깅 비활성화
wandb_project = 'gpt-2-pretraining' # W&B 프로젝트명 설정
wandb_run_name = 'gpt-2-pretraining' # 실행 이름을 'gpt2'로 설정

# 학습 데이터셋
# Gradient Accumulation: GPU 메모리 한계를 극복하기 위해 작은 배치를 여러번 누적해서 학습하는 방법
dataset = 'openwebtext' # 사용할 데이터셋 이름 넣기
gradient_accumulation_steps = 5 * 8 # gradient accumulation을 사용해 batch size 증가
batch_size = 8 # micro batch 사이즈 크기 설정
block_size = 64 # max sequence length (컨텍스트 길이, 최대 토큰 수) 설정

# 모델 설정
n_layer = 4 # transformer의 레이어 수
n_head = 4 # multi-head attention에서 head 개수
n_embd = 128 # 임베딩의 차원 크기
dropout = 0.0 # 사전학습에서는 0을 사용, fine-tuning 할 때는 0.1+을 추천
bias = False # LayerNorm and Linear layers에서 bias 사용할지 유무

# AdamW optimizer 설정
learning_rate = 6e-4 # max learning rate 설정
max_iters = 2000 # 총 training iteration 수 (최대 학습 스텝 수)
max_iters = 2000 # 총 training iteration 수 (최대 학습 스텝 수)
weight_decay = 1e-1 # weight decay 값 설정
beta1 = 0.9 # AdamW의 모멘텀 계수
beta2 = 0.95 # AdamW의 모멘텀 계수
grad_clip = 1.0 # gradient 폭발 방지 장치: gradient clipping 기준값 설정 0 이면 disable 됨
# gradient clipping: 발산하지 않도록 일정값에서 saturation 시키는 장치

# learning rate decay 설정
# Cosine Decay: learning rate를 점진적으로 감소시키는 방식 (cosine 함수 사용)
# learning rate warmup 어디까지 해야하나요? 질문 > 답 할 수 있어야함
decay_lr = True # learning rate를 decay 할지 여부 설정
warmup_iters = 2000 # warm up할 스텝 수 설정 / 꺾이는 시점 / # https://sonseungha.tistory.com/720
lr_decay_iters = 2000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# Distributed Data Parallel(DDP) 데이터 분산 방법 설정
# Data Parallelism(어떻게 배치를 나눌 것인가), 3가지 parallelism 알아야 함
# https://www.notion.so/jwkangmarco/3D-parallelism-882c34a9e746484d88010dedc93fc399
# 백엔드 설정
# 'nccl': NVIDIA Collective Communications Library: 멀티 GPU 통신 최적화
# 'gloo':
backend = 'nccl' # 'nccl': gpu 환경, 'gloo': cpu 환경

# cpu, gpu, mps 중 선택
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

# 'bfloat16' vs 'float16' 선택 (float32도 있음: 메모리가 부족): 1 parameter = 4 byte 여서 메모리를 더 많이 차지함, 정확도는 더 높음
# 16 정도도 precision 괜찮다
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # PyTorch 2.0 컴파일 사용 여부


# configurator.py 에서 config keys와 config 값들 가져오기
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


wandb_config = {
    "tokenizer_max_length": block_size,
    "learning_rate": learning_rate,
    "batch_size": batch_size
}

wandb.init(config=wandb_config, project="gpt-2-pretraining", entity="llm-research-adena-new-york-university")




# DDP 설정
# DDP 모드 실행 여부 확인
# os.environ.get('RANK', -1): 환경변수에서 'RANK' 값을 가져옴
# 만약 RANK 값이 설정되지 않았다면 기본값으로 -1을 반환함
# RANK 값이 -1이 아니면 DDP 모드가 활성화된 상태를 의미
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

# RANK: 전체 분산학습에서 현재 프로세스의 순서를 의미함
# 예를 들어, ddp_world_size = 4 인 경우, RANK = 0, 1, 2, 3 중 하나가 됨

# DDP가 활성화되었을 경우:
if ddp:
    # DDP 프로세스 그룹을 초기화 및 설정
    init_process_group(backend=backend) # backend='nccl': gpu, 'gloo': cpu 둘 중 하나 선택

    ddp_rank = int(os.environ['RANK']) # 현재 프로세스의 RANK
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # 현재 노드 내에서 사용중인 GPU 번호
    # ex) 4개의 gpu가 있는 머신일 경우 LOCAL_RANK=0,1,2,3 중 하나가 됨

    ddp_world_size = int(os.environ['WORLD_SIZE']) # 전체 학습에 참여하는 프로세스 개수 ex) 1
    # ex) GPU 4개에서 2개의 노드를 사용할 경우, WORLD_SIZE=8이 됨 (보통 노드 1개 = GPU 8개)

    # 학습할 디바이스 및 gpu 설정: 각 프로세스가 사용할 gpu를 지정함
    # ex) LOCAL_RANK=2 이면, 해당 프로세스는 cuda:2에서 실행이 됨
    device = f'cuda:{ddp_local_rank}' # 현재 프로세스가 사용할 gpu 선택
    torch.cuda.set_device(device) # PyTorch에서 해당 gpu를 기본 장치로 설정

    # master 프로세스 확인: RANK가 0이면 master 프로세스임
    # master 프로세스는 로그를 기록하고, 체크포인트를 저장하는 역할을 담당함
    master_process = ddp_rank == 0 # RANK가 0인 프로세스를 찾아서 저장함

    # Random Seed 설정
    # 동일한 데이터를 학습하는 것을 방지하기 위해 random seed offset 설정하여 프로세스마다 다른 seed값을 부여
    seed_offset = ddp_rank # 각 프로세스가 다른 seed를 가지도록 설정

    # Gradient Accumulation(그래디언트 누적: 스케줄링()): gpu 메모리 부족을 해결하기 위해 사용됨 > 한번에 배치를 올리는 메모리를 줄이기 위해 / 단점: 느려질수있다
    # DDP 환경에서는 각 프로세스가 일부 데이터만 처리하므로 전체 batch size를 조정해야함
    # gradient_accumulation_steps 값을 전체 프로세스 수인 ddp_world_size로 나누어 균등하게 분배해야됨
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size # 똑같이 분배하도록 할당할 데이터 크기 구하기

else:
    # DDP가 비활성화 된 경우: gpu 1개(프로세스 1개)로 학습하는 경우
    master_process = True # 단일 프로세스일 경우 항상 master 프로세스가 됨
    seed_offset = 0 # random seed 초기화
    ddp_world_size = 1 # DDP가 아니므로 world_size(전체개수)는 1이 됨

# 전체 batch 크기 계산
# 한 번의 iteration에서 처리 할 토큰 수를 계산
# gradient_accumulation_steps: Accumulation 단계를 고려한 배치 크기
# ddp_world_size: 전체 프로세스 개수
# batch_size: 각 프로세스에서 처리하는 미니 배치 크기
# block_size: 최대 문장 길이 (토큰 수)
# 중요 > 필요한 메모리를 결정함
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size # block size: max sequence length
# 최종적으로 한번의 학습 반복에서 몇 개의 토큰을 처리하는지 출력함
print(f"tokens per iteration will be: {tokens_per_iter:,}")


# master 프로세스 설정 및 random seed 초기화
# master 프로세스만 out_dir 폴더를 생성하여 모델 체크포인트 등을 저장하도록 함

# exit_ok=True 옵션을 사용해서 폴더가 이미 존재해도 오류가 발생하지 않도록 함
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Random Seed 설정: 1337은 기본값임
# 앞에서 만든 seed_offset을 1337에 더해줌
torch.manual_seed(1337 + seed_offset)

# TensorFloat-32(TF32) 활성화 > NVIDIA gpu의 core (요즘 gpu에 있음) / NVIDIA gpu 구조를 알 필요가 있음 (nanoGPT)
# NVIDIA gpu spec에 대해 알고 있는게 좋음
# A100, V100, H100 > 모델 학습용

# TF32는 NVIDIA gpu에서 사용하는 속도 최적화된 연산 형식임
# TF32를 활성화하면 학습 속도가 향상되지만, 일부 경우 정확도가 낮아질 수 있음
# 최신 gpu(Ampere:A100 이상, RTX30 시리즈 이상(30,40,..))에서는 TF32를 활성화하면 연산 성능이 최대 8배 증가함
torch.backends.cuda.matmul.allow_tf32 = True # 행렬 연산에서 TF32 활성화시킴 allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # CuDNN에서 TF32 활성화시킴

# 학습 디바이스(gpu vs cpu) 설정
device_type = 'cuda' if 'cuda' in device else 'cpu' # 'cuda'이면 gpu 사용 아니면 cpu 사용, mps: 맥북 gpu

# dtype 데이터 타입 설정 (PyTorch 데이터 타입: float16, bfloat16, float32 중 하나 사용)
# float16 타입은 자동적으로 GradScaler를 사용함
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# autocast를 사용하여 자동 혼합 정밀도 연산(AMP, Automatic Mixed Precision)을 활성화함 > dtype을 섞어서 사용하게 함
# cpu 사용시 nullcontext()를 사용하여 아무 작업도 하지 않음
# gpu 사용시 torch.amp.autocast()를 사용하여 bfloat16 또는 float16을 자동으로 적용함
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



# 학습데이터셋 경로 설정
data_dir = os.path.join('data', dataset) # 데이터셋이 저장된 폴더 넣기: 'data/openwebtext'

# We recreate np.memmap every batch to avoid a memory leak, as per
# https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

# 데이터 로딩 함수
def get_batch(split):
    # split이 'train'이면 train 데이터를 로딩, 'val'이면 validation 데이터를 로딩함
    # 메모리 매핑(memory mapping=memmap) 방식으로 바이너리(bin) 데이터를 읽음: 메모리 효율적인 데이터 로딩이 가능함
    # train.bin 또는 val.bin을 불러옴
    # dtype=np.uint16 (unsigned int 16) 16비트 정수 형태로 저장된 데이터를 읽음
    # mode='r': 읽기 전용 모드
    # numpy.memmap을 매 배치마다 새롭게 생성하여 메모리 누수를 방지함

    if split == 'train':
        # numpy.memmap을 매 배치마다 새롭게 생성하여 메모리 누수를 방지함
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        # numpy.memmap을 매 배치마다 새롭게 생성하여 메모리 누수를 방지함
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # batch_size 크기만큼 random한 시작 위치를 선택
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # next token prediction: x로 -> y 예측 (y값을 한 칸 미루는 이유)
    # shape을 알아야함
    # x: block_size 길이만큼의 입력 시퀀스를 생성 ex) [0,1,2,...]]
    # y: x 보다 한 칸 오른쪽으로 밀린 정답 데이터 ex) [1,2,3,...]]
    # 데이터는 int ID(np.int64)로 변환하여 tensor로 변환함
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])


    # 디바이스 종류
    if device_type == 'cuda': # gpu 사용시
        # .pin_memory(): 데이터를 고정 메모리에 저장하여 비동기 전송을 가능하게 함 pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        # .to(device, non_blocking=True): cpu -> gpu 비동기 전송을 수행하여 속도를 높임
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else: # cpu 사용시
        x, y = x.to(device), y.to(device)

    # 최종 x: 입력데이터, y: 정답데이터 리턴
    return x, y



# 학습 체크포인트 초기화
iter_num = 0 # 현재 iteration 횟수 초기화
best_val_loss = 1e9 # 초기 best loss를 매우 큰 값(무한∞)으로 설정


# 데이터셋에서 vocab_size 추출
meta_path = os.path.join(data_dir, 'meta.pkl') # meta.pkl 파일이 존재하면 vocab_size를 추출
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f) # 바이너리 데이터를 파이썬 객체로 로드
    meta_vocab_size = meta['vocab_size'] # 토크나이저에서 사용한 vocab_size를 가져옴
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})") # vocab_size 출력


# 모델 hyperparameter 설정
# 모델을 초기화하기 위해 필요한 hyperparameter 값을 model_args 딕셔너리에 저장
# n_layer: transformer의 블록 개수
# n_head: multi-head attention 개수
# n_embd: 임베딩 차원 크기
# block_size: 입력 시퀀스 최대 길이
# bias: LayerNorm과 Linear layer에서 bias 사용 여부
# vocab_size: 어휘 크기, 초기값: None
# dropout: 드롭아웃 비율

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

# 모델 초기화 방식
# init_from 값을 아래 3가지 중 1개 선택
# 'scratch': 모델을 새로 생성
# 'resume': 체크포인트에서 재개
# 'gpt2': GPT-2 사전학습 가중치 불러오기

if init_from == 'scratch': # 1. 처음부터 새로운 모델 생성
    print("Initializing a new model from scratch")
    if meta_vocab_size is None: # meta.pkl 파일에 vocab_size 정보가 없으면 GPT-2의 기본 vocab_size를 50257 > 50304로 설정
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

    # GPTConfig(**model_args)를 사용해 모델 설정을 초기화하고 GPT(gptconf)로 모델을 생성함
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)


elif init_from == 'resume': # 2. 체크포인트에서 학습 재개
    print(f"Resuming training from {out_dir}")
    # 기존 학습을 이어서 진행하는 경우, 저장된 체크포인트(ckpt.pt)를 볼러옴
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # 기존 체크포인트에서 모델 설정값을 가져와 일관성을 유지함
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    # 불러온 설정값으로 새로운 GPT 모델을 생성!
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # 모델 가중치 불러오기
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.' # 불필요한 접두사가 포함된 경우 제거 적용
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # 체크포인트에서 이전 학습 상태(반복 횟수, 최적 loss값)를 복원함
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']


elif init_from.startswith('gpt2'): # 3.GPT-2 사전학습 가중치 불러오기
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # GPT-2 모델을 불러와서 학습을 진행
    # dropout 값을 현재 설정값으로 덮어씀
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)

    # 불러온 GPT-2 모델에서 hyperparameter를 가져와 설정을 업데이트함
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)


# 모델 블록 크기 조정
# 모델의 block_size가 학습 설정값보다 크다면 모델 구조를 수정하여 크기를 줄여야함
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# PyTorch AMP(Automatic Mixed Precision) 설정
# 자동 혼합 정밀도 사용 여부를 결정
# float16을 사용할 경우, GradScaler를 사용하여 정밀도를 유지하면서 성능을 최적화함
# GradScaler 초기화 / If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer 설정
# 모델에 최적화된 AdamW Optimizer를 적용함
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# 학습을 재개(resume)할 경우, 체크포인트에서 옵티마이저 상태를 복원함
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory 메모리 비우기

# 모델 컴파일 (PyTorch 2.0)
# PyTorch 2.0의 torch.compile()을 사용하여 모델을 컴파일 및 최적화함
# 속도를 향상시키지만, 첫 실행에 시간이 더 걸릴 수 있음
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
# DDP container에 모델 넣기
# DDP(multi GPU 학습) 환경이라면 모델을 DDP 컨테이너로 래핑해야함
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
# 모델을 평가할 때는 역전파가 필요하지 않기 때문에 연산량을 줄여 메모리를 절약함
@torch.no_grad() # torch.no_grad() 데코레이터를 사용하여 gradient 계산을 비활성화함
def estimate_loss():
    out = {}
    model.eval() # 학습중이 아닌 평가모드로 변경 / Dropout, BatchNorm 등의 레이어가 평가 모드로 전환됨
    for split in ['train', 'val']:
        # train과 val 데이터셋에 대해 각각 loss를 평가함
        # 평가 반복 횟수(eval_iters)만큼 loss값을 저장할 tensor를 생성함
        losses = torch.zeros(eval_iters) # 손실값 저장을 위한 tensor 초기화

        # 여러 배치를 가져와서 평균 loss를 계산함
        # ctx는 AMP(자동 혼합 정밀도, Mixed Precision)를 활성화는 컨텍스트임
        # float16 또는 bfloat16 연산을 사용하여 메모리 사용량을 줄이고 속도를 높임
        for k in range(eval_iters):
            # 여러 배치르 가져
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()

        # 평균 loss값 저장
        out[split] = losses.mean()

        # W&B 로깅 (평가 손실 추가)
        wandb.log({f"{split}_eval_loss": out[split]})

    # 평가가 끝나면 모델을 다시 학습 모드로 변경
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
# learning rate 스케줄러 (get_lr)
# 현재 학습 반복 횟수(it)에 따라 learning rate를 결정하는 스케줄러 함수
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    # 초기 warmup_iters 스텝 동안 learning rate를 선형적으로 증가시킴
    # 작은 learning rate에서 시작해서 점진적으로 증가시키면 학습이 안정적으로 진행됨
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    # 2) if it > lr_decay_iters, return min learning rate
    # 학습이 일정 단계(lr_decay_iters) 이상 진행되면 최소 learning rate(min_lr)로 고정
    if it > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    # Cosine Decay 방식을 사용하여 learning rate를 점진적으로 감소시킴
    # 처음에는 빠르게 감소하다가 후반부로 갈수록 천천히 감소하는 형태가 됨
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 코사인 감소
    return min_lr + coeff * (learning_rate - min_lr)


# logging 설정(Weights & Biases)
# wandb_log가 활성화된 경우 Weights & Biases 로깅을 초기화함
# master 프로세스만 로깅을 수행하여 불필요한 중복을 방지
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# 학습 루프 (while True)
# 처음 학습을 시작하기 전에 첫번째 배치를 가져옴
# multi-GPU(DDP) 학습 환경에서는 model.module을 사용하여 원본 모델에 접근
X, Y = get_batch('train') # 첫번째 배치 가져오기
t0 = time.time() # 학습 시작 시간 저장
local_iter_num = 0 # 현재 프로세스의 학습 반복 횟수
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0 # gpu 사용량 추적 / mfu: gpu를 얼마나 효율적으로 사용하고있는지 알려주는 장치

while True:

    # determine and set the learning rate for this iteration
    # learning rate 설정
    # get_lr(iter_num)을 호출하여 현재 iteration에 대한 learning rate를 설정함
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 일정 간격(eval_interval) 마다 train 및 validation loss를 평가함
    # 새로운 최적의 validation loss가 나오면 체크포인트(ckpt.pt)를 저장함
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
         # 새로운 최적의 validation loss가 나오면 체크포인트(ckpt.pt)를 저장함
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16

    # 순전파 & 역전파 (Forward & Backward)
    # Gradient Accumulation을 수행하여 더 큰 배치 크기를 시뮬레이션함
    # float16 학습 시 gradient를 스케일링(재조정)하여 underflow 방지(너무 작은 값이 나오는 것)
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # gradient 누적을 고려하여 loss scaling / scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    # Gradient 업데이트
    # clip_grad_norm을 적용하여 Gradient Explosion(폭발) 방지
    # scaler.step(optimizer), scaler.update()를 사용하여 혼합 정밀도 학습 적용
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer) # backpropagation 시작하는 부분
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # 로그 출력 및 학습 반복
    # loss 값 및 학습 속도를 로그로 출려함
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")



    iter_num += 1
    local_iter_num += 1

    # 학습 종료 조건
    # max_iters 이상 반복되면 학습을 종료함
    if iter_num > max_iters:
        break

# 학습 종료 및 남은 것 다 종료
# DDP가 활성 되어있는 경우 destroy_process_group()을 사용해서 분산 학습 프로세스 그룹 종료
if ddp:
    destroy_process_group()

# 학습 종료 후 W&B 종료
if wandb_log:
    wandb.finish()
