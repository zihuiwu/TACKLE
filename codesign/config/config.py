import importlib, pathlib, torch
from fvcore.common.config import CfgNode as CN
from codesign.engines import *

class CodesignConfigurator:
    def __init__(self, args):
        """
        Create configs and make fixes
        """
        self.cfg = CN(CN.load_yaml_with_base(args.config))

        # check cfg configurations
        self._check_cfg()

        # user provided fixes
        if args.fix is not None:
            self.cfg.merge_from_list(args.fix)

        # default fixes
        self.cfg = self._default_fix(self.cfg)
        
        self.cfg.freeze()
    
    def _check_cfg(self):
        assert self.cfg.data_module.shape == self.cfg.model.sampler.shape, 'data shape and sampler shape are inconsistent!'
        if 'init_modules' in self.cfg.keys():
            assert set(self.cfg.init_modules) <= set(['sampler', 'reconstructor', 'predictor']), 'init_modules must be a subset of {"sampler", "reconstructor", "predictor"}!'

    def _default_fix(self, cfg):
        cfg.exp_dir =  f"results/{cfg.exp_name}"
        cfg.logger.name = cfg.exp_name
        cfg.logger.save_dir = cfg.exp_dir
        cfg.trainer.default_root_dir = cfg.exp_dir
        return cfg

    @staticmethod
    def str_to_class(module_name, class_name):
        """Return a class instance from a string reference"""
        try:
            module_ = importlib.import_module(module_name)
            try:
                class_ = getattr(module_, class_name)
            except AttributeError:
                raise AttributeError('Class does not exist')
        except ImportError:
            raise ImportError('Module does not exist')
        return class_

    @staticmethod
    def init_params_without_name(module_name, cfg):
        class_ = CodesignConfigurator.str_to_class(module_name, cfg.name)
        init_dict = dict(cfg)
        del init_dict["name"]
        return class_(**init_dict)

    def _init_data_module(self):
        return self.init_params_without_name("codesign.data", self.cfg.data_module)

    def _init_sampler(self):
        return self.init_params_without_name("codesign.samplers", self.cfg.model.sampler)
    
    def _init_reconstructor(self):
        return self.init_params_without_name("codesign.reconstructors", self.cfg.model.reconstructor)

    def _init_predictor(self):
        return self.init_params_without_name("codesign.predictors", self.cfg.model.predictor)
    
    def _init_train_loss(self):
        return self.init_params_without_name("codesign.losses", self.cfg.train_loss)
    
    def _init_val_test_loss(self):
        return self.init_params_without_name("codesign.losses", self.cfg.val_test_loss)

    def _init_exp(self):
        if self.cfg.procedure.name == 'TrainValTest':
            exp = TrainValTest(self.cfg)
        elif self.cfg.procedure.name == 'CrossValidation':
            exp = CrossValidation(self.cfg, self.cfg.procedure.num_folds)
        else:
            raise NotImplementedError
        return exp

    def _init_ckpt(self, exp_dir):
        ckpt_list = []
        for fname in list(pathlib.Path(exp_dir).iterdir()):
            if fname.name[-5:] == '.ckpt':
                ckpt_list.append(fname)
        
        if len(ckpt_list) == 0:
            raise FileNotFoundError('There is no checkpoint in the directory!')
        elif len(ckpt_list) > 1:
            raise RuntimeError('There are multiple checkpoints in the directory!')

        return ckpt_list[0]

    def _load_model_from_ckpt(self, ckpt, model, ignore_train_loss=True, ignore_val_test_loss=True):
        # state dict of the model for initialization
        init_model_dict = torch.load(ckpt)['state_dict']
        
        # state dict of the model to be initialized
        model_dict = model.state_dict()

        # filter out unrelated parameters
        filtered_init_model_dict = {}
        for k, v in init_model_dict.items():
            k_module = k.split('.')[0]
            if k in model_dict and k_module in self.cfg.init_modules:
                filtered_init_model_dict[k] = v

        # initialize
        model_dict.update(filtered_init_model_dict) 
        model.load_state_dict(model_dict)

        # set trainability of initialized modules
        for m, t in zip(self.cfg.init_modules, self.cfg.init_module_trainability):
            for param in model.__getattr__(m).parameters():
                param.requires_grad = t

        return model

    def _init_model(self):
        sampler = self._init_sampler()
        reconstructor = self._init_reconstructor()
        predictor = self._init_predictor()
        train_loss = self._init_train_loss()
        val_test_loss = self._init_val_test_loss()

        if self.cfg.procedure.name == 'TrainValTest':
            model = LitCoDesign(
                self.cfg, 
                sampler, 
                reconstructor, 
                predictor, 
                train_loss, 
                val_test_loss
            )
        elif self.cfg.procedure.name == 'CrossValidation':
            model = LitCrossValidationCoDesign(
                self.cfg, 
                sampler, 
                reconstructor, 
                predictor, 
                train_loss, 
                val_test_loss
            )
        else:
            raise NotImplementedError

        # assert model.task == model.train_loss.task, 'the task specified for the model is inconsistent with the training loss'

        return model

    def init_all(self):
        # model
        model = self._init_model()

        # checkpoint
        if 'init_exp_dir' in self.cfg.keys():
            ckpt = self._init_ckpt(self.cfg.init_exp_dir)
            model = self._load_model_from_ckpt(ckpt, model, ignore_train_loss=True, ignore_val_test_loss=True)
        
        # exp (TrainValTest or CrossValidation)
        exp = self._init_exp()

        # data module
        data_module = self._init_data_module()

        return exp, model, data_module

class CodesignTestConfigurator(CodesignConfigurator):
    def __init__(self, args):
        """
        Create configs and make fixes
        """
        super().__init__(args)
        self.data_cfg = CN(CN.load_yaml_with_base(args.data_config))
        
        # check cfg configurations
        self._check_data_cfg()

        self.data_cfg.freeze()

    def _check_data_cfg(self):
        if self.cfg.model.sampler.shape != self.data_cfg.data_module.shape:
            raise RuntimeError(f'The shape of the sampler ({self.cfg.model.sampler.shape}) does not match the shape of the data module to be tested ({self.data_cfg.data_module.shape})!')

        if self.data_cfg.data_module.name == 'MartinosPathoDataModule':
            pass
            # if self.cfg.data_module.pathologies != [0]:
            #     raise RuntimeError('Model is not trained with pathologies = [0] but tested on MartinosPathoDataModule!')
        
        if self.data_cfg.data_module.name is not self.cfg.data_module.name:
            pass
            # raise RuntimeError(f'Model trained with {self.cfg.data_module.name} but tested on {self.data_cfg.data_module.name}.' \
            #                     ' Append "-f [data module configurations] [loss configurations]" at the end of the command line!')

    def _init_data_module(self):
        return self.init_params_without_name("codesign.data", self.data_cfg.data_module)
    
    def _init_val_test_loss(self):
        return self.init_params_without_name("codesign.losses", self.data_cfg.val_test_loss)

    def _load_model_from_ckpt(self, ckpt, model, ignore_train_loss=True, ignore_val_test_loss=True):
        # record train_loss & val_test_loss 
        if ignore_train_loss:
            train_loss = model.train_loss
        if ignore_val_test_loss:
            val_test_loss = model.val_test_loss

        # load model
        model.load_state_dict(torch.load(ckpt)['state_dict'], strict=False)

        # assign train_loss & val_test_loss to the recorded type
        if ignore_train_loss:
            model.train_loss = train_loss
        if ignore_val_test_loss:
            model.val_test_loss = val_test_loss
        
        return model

    def init_all(self):
        # model
        model = self._init_model()

        # checkpoint
        ckpt = self._init_ckpt(self.cfg.exp_dir)
        model = self._load_model_from_ckpt(ckpt, model, ignore_train_loss=True, ignore_val_test_loss=True)
        
        # exp
        exp = Test(self.cfg, self.data_cfg)

        # data module
        data_module = self._init_data_module()

        return exp, model, data_module  