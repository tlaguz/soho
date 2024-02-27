from dataclasses import dataclass


@dataclass
class TrainingMetadataDto:
    time: str
    time_spent: float
    local_rank: int
    local_device: str
    epoch: int
    iteration: int
    epoch_loss: float
    running_loss: float
    loss: float
    valid_loss: float
    checkpoint_tag: str
    lr: float = 0.0

# used to read and save training metadata
# file contains one dataclass object per line
# each object is a TrainingMetadataDto
class TrainingMetadata:
    def __init__(self, metadata_file):
        self.metadata_file = metadata_file

    def read(self):
        with open(self.metadata_file, "r") as f:
            return [eval(line) for line in f]

    def save(self, metadata: [TrainingMetadataDto]):
        with open(self.metadata_file, "w") as f:
            for line in metadata:
                f.write(str(line) + "\n")

    def append(self, metadata: TrainingMetadataDto):
        with open(self.metadata_file, "a") as f:
            f.write(str(metadata) + "\n")
