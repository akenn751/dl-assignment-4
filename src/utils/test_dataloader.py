from src.utils.dataloader import SequenceDataLoader

loader = SequenceDataLoader(mode="char", seq_length=10)

print("Dataset length:", len(loader))
inp, tgt = loader.get_sequence(0)
print("Input:", inp)
print("Target:", tgt)

for batch_inputs, batch_targets in loader.batch(4):
    print("Batch inputs shape:", batch_inputs.shape)
    print("Batch targets shape:", batch_targets.shape)
    break
