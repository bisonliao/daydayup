# frap_conf.py
import torch

class Config:
    # geometry / problem
    PHASE_NUM = 4        # 4 相位：0 NS-through, 1 EW-through, 2 NS-left, 3 EW-left
    MOVEMENT_NUM = 8     # movements: N_through, N_left, E_through, E_left, S_through, S_left, W_through, W_left
    FEAT_DIM = 1         # the length of movement's feature

    # mapping phase -> movement indices (based on above movement order)
    # phase 0 (NS-through) uses movements 0 (N_through) and 4 (S_through)
    # phase 1 (EW-through) uses movements 2 (E_through) and 6 (W_through)
    # phase 2 (NS-left)    uses movements 1 (N_left)    and 5 (S_left)
    # phase 3 (EW-left)    uses movements 3 (E_left)    and 7 (W_left)
    '''
    N_through (0) = 从北边驶入 → 往南直行
    S_through (4) = 从南边驶入 → 往北直行
    E_through (2) = 从东边驶入 → 往西直行
    W_through (6) = 从西边驶入 → 往东直行
    N_left (1) = 从北边驶入 → 往东左转
    S_left (5) = 从南边驶入 → 往西左转
    E_left (3) = 从东边驶入 → 往南左转
    W_left (7) = 从西边驶入 → 往北左转
    '''
    PHASE_MOVEMENTS = [
        [2, 6],
        [0, 4],
        [3, 7],
        [1, 5],
    ]

    # 网络维度配置
    DEMAND_HIDDEN_DIM = 32
    DEMAND_OUTPUT_DIM = 16
    RELATION_EMBEDDING_DIM = 16
    DEMAND_EMBEDDING_DIM = 32
    COMPETITION_HIDDEN_DIM = 64
    COMPETITION_LAYERS = 2
    RELATION_LAYERS = 2
     

    # training / DQN
    GAMMA = 0.9
    LR = 1e-3
    BATCH_SIZE = 128
    BUFFER_SIZE = int(1e6)
    MIN_REPLAY_SIZE = BATCH_SIZE*3
    TARGET_UPDATE_FREQ = 2000    # steps
    TRAIN_FREQ = 1               # how often to do a gradient step (in env steps)
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY_RATE = 0.98
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # other runtime
    SEED = 31
    PRINT_EVERY = 1

    CELL_LEN=5
    OBSERV_CELL_NUM=6
    MAX_QLEN=30
    MAX_PHASE_DUR=30
    LANE_LEN = 300
    ENV_BASE_DIR = '/home/bison/tsc/FRAP/env'
