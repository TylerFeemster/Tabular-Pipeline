from utils import gpu_available
from typing import Union, Tuple
from optuna import Trial

### Implemented Models

MODELS = [
    'hist_long',
    'hist_wide',
    'prox_long',
    'prox_wide',
    'rndm_long',
    'rndm_wide',
    'linr'
]

### Model-Defining Parameters

CONSTRUCTOR_PARAMS = {
    'hist_long': {
        'tree_method': 'hist',
        'grow_policy': 'lossguide'
    },
    'hist_wide': {
        'tree_method': 'hist',
        'grow_policy': 'depthwise'
    },
    'prox_long': {
        'tree_method': 'approx',
        'grow_policy': 'lossguide'
    },
    'prox_wide': {
        'tree_method': 'approx',
        'grow_policy': 'depthwise'
    },
    'rndm_long': {
        'learning_rate': 1,
        'grow_policy': 'lossguide'
    },
    'rndm_wide': {
        'learning_rate': 1,
        'grow_policy': 'depthwise'
    },
    'linr': {
        'booster': 'gblinear'
    }
}

### GPU/CPU Parameters

COMPUTE_PARAMS = {
    'gpu': {
        'device': 'gpu'
    },
    'cuda': {
        'device': 'cuda'
    },
    'cpu': {
        'device': 'cpu'
    }
}

### Hyperparameters

# 'param': (min of range, max of range, float?, log?)

REG_PARAMS = {
    'lambda': (1e-1, 1e3, True, True),
    'alpha': (1e-10, 1, True, True)
}

TREE_PARAMS = {
    'max_bin': (128, 512, False, True),
    'subsample': (1e-1, 1, True, True),
    'colsample_bynode': (5e-1, 1, True, True),
    **REG_PARAMS
}

BOOST_PARAMS = {
    'learning_rate': (5e-3, 5e-1, True, True),
    **TREE_PARAMS
}

RF_PARAMS = {
    'num_parallel_tree': (100, 4000, False, True),
    **TREE_PARAMS
}

LONG_PARAMS = {
    'max_leaves': (16, 512, False, True)
}

WIDE_PARAMS = {
    'max_depth': (4, 9, False, False)
}

HYPER_PARAMS = {
    'hist_long': {
        **BOOST_PARAMS,
        **LONG_PARAMS
    },
    'hist_wide': {
        **BOOST_PARAMS,
        **WIDE_PARAMS
    },
    'prox_long': {
        **BOOST_PARAMS,
        **LONG_PARAMS
    },
    'prox_wide': {
        **BOOST_PARAMS,
        **WIDE_PARAMS
    },
    'rndm_long': {
        **RF_PARAMS,
        **LONG_PARAMS
    },
    'rndm_wide': {
        **RF_PARAMS,
        **WIDE_PARAMS
    },
    'linr': {
        **REG_PARAMS
    }
}

### Toy Parameters

TOY_REG_PARAMS = {
    'lambda': 1,
    'alpha': 0
}

TOY_TREE_PARAMS = {
    'max_bin': 128,
    'subsample': 1,
    'colsample_bynode': 1,
    **TOY_REG_PARAMS
}

TOY_BOOST_PARAMS = {
    'learning_rate': 5e-2,
    **TOY_TREE_PARAMS
}

TOY_RF_PARAMS = {
    'num_parallel_tree': 500,
    **TOY_TREE_PARAMS
}

TOY_LONG_PARAMS = {
    'max_leaves': 32
}

TOY_WIDE_PARAMS = {
    'max_depth': 5
}

TOY_PARAMS = {
    'hist_long': {
        **TOY_BOOST_PARAMS,
        **TOY_LONG_PARAMS
    },
    'hist_wide': {
        **TOY_BOOST_PARAMS,
        **TOY_WIDE_PARAMS
    },
    'prox_long': {
        **TOY_BOOST_PARAMS,
        **TOY_LONG_PARAMS
    },
    'prox_wide': {
        **TOY_BOOST_PARAMS,
        **TOY_WIDE_PARAMS
    },
    'rndm_long': {
        **TOY_RF_PARAMS,
        **TOY_LONG_PARAMS
    },
    'rndm_wide': {
        **TOY_RF_PARAMS,
        **TOY_WIDE_PARAMS
    },
    'linr': {
        **TOY_REG_PARAMS
    }
}

### Boostable

BOOSTABLILITY = {
    'hist_long': True,
    'hist_wide': True,
    'prox_long': True,
    'prox_wide': True,
    'rndm_long': False,
    'rndm_wide': False,
    'linr': False
}

TOY_BOOST = 500
BOOST_HYPER_PARAM = (1e2, 5e3, False, True)

class ModelInfo:

    def __init__(self, model : str, device : Union[str, None] = None, 
                 objective : Union[str, None] = None,
                 seed : int = 0) -> None:

        assert model in MODELS, f"model must be one of {MODELS}."
        self.model = model

        # Device
        if device:
            assert device in ['cpu', 'cuda', 'gpu'], \
                "device must be \'cpu\', \'cuda\', or \'gpu\'"
            self.device = device
        elif gpu_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Criterion
        objective_params = dict()
        if objective:
            objective_params['objective'] = objective
        
        self.main_params = {**CONSTRUCTOR_PARAMS[model],
                            **COMPUTE_PARAMS[self.device],
                            **objective_params}
        self.main_params['seed'] = seed

        self.hyper_params = HYPER_PARAMS[model]
        self.toy_params = TOY_PARAMS[model]

        self.is_boostable = BOOSTABLILITY[model]

        self.best_params = dict()
        self.best_boost = None
        
        return
    
    def minimal_parameters(self) -> Tuple[dict, int]:
        if self.is_boostable:
            num_boost_round = TOY_BOOST
        else:
            num_boost_round = 1

        parameters = {**self.main_params}

        return parameters, num_boost_round

    def feature_selection_parameters(self) -> Tuple[dict, int]:
        if self.is_boostable:
            num_boost_round = TOY_BOOST
        else:
            num_boost_round = 1

        parameters = {**self.main_params,
                      **self.toy_params}

        return parameters, num_boost_round
    
    def trial_parameters(self, trial : Trial) -> Tuple[dict, int]:
        parameters = {**self.main_params} # copy
        for name, (min, max, is_float, do_log) in self.hyper_params.items():
            if is_float:
                parameters[name] = trial.suggest_float(name, min, max, log=do_log)
            else:
                parameters[name] = trial.suggest_int(name, min, max, log=do_log)
        
        if self.is_boostable:
            min, max, is_float, do_log = BOOST_HYPER_PARAM
            num_boost_round = trial.suggest_int('num_boost_round', min, max, log=do_log)
        else:
            num_boost_round = 1

        return parameters, num_boost_round
    
    def set_best_parameters(self, params : dict) -> None:

        for key, item in params.items():
            if key == 'num_boost_round':
                self.best_boost = item
            else:
                self.best_params[key] = item

        return

        
    def get_best_parameters(self) -> Tuple[dict, int]:

        if self.is_boostable:
            assert self.best_boost is not None
            num_boost_round = self.best_boost
        else:
            num_boost_round = 1

        parameters = {**self.main_params,
                      **self.best_params}
        
        return parameters, num_boost_round

