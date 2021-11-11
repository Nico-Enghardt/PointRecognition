import wandb
api = wandb.Api()
run = api.run("nico-enghardt/PointRecognition/3uzu3pxj")
run.config["trainingExamples"] = 5700
run.update()