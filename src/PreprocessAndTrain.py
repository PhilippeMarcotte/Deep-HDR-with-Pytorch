from PrepareTrainingData import distribute_training_data_preparation
from TrainersDeepHDR import DirectTrainerDeepHDR

distribute_training_data_preparation()
trainer = DirectTrainerDeepHDR()
trainer.train()
