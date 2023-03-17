import torch.utils.dlpack
from pyhie import allspark
from rlhf.model_wrapper import RLHFModule


class GPTAllSpark(RLHFModule):

    def setup(self):
        engine = allspark.Engine()
        engine.set_device_type("CUDA")
        device_ids = list(range(self.num_device))
        engine.set_device_ids(device_ids)
        model_path = self.model_args['model_path']
        torch_model = torch.load(model_path)
        model_config = self.model_args["model_config"]
        model_config["layer_norm_eps"] = float(model_config["layer_norm_eps"])
        model_config["layernorm_epsilon"] = float(model_config["layernorm_epsilon"])
        self.generate_config = self.model_args["generate_config"]
        print(model_config)

        engine.build_model_from_torch(
            model_name=self.name,
            model_type="GPT3",
            torch_model=torch_model,
            data_type="float16",
            model_config=model_config,
            derive_type="lmhead",
        )
        self.engine = engine
        return 'ok'


    def forward_step(self, data):
        out = self.engine.run_text_generation(
            self.name,
            {
                "input_ids": torch.utils.dlpack.to_dlpack(data["input_ids"]),
                "attention_mask": torch.utils.dlpack.to_dlpack(data["attention_mask"]),
            },
            self.generate_config,
        )
        response_tensor = torch.utils.dlpack.from_dlpack(out["generated_ids"])
        return {"response": response_tensor}
