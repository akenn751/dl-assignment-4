from src.utils.tokenizer import CharTokenizer, WordTokenizer

ct = CharTokenizer()
print(ct.encode("Hello"))
print(ct.decode(ct.encode("Hello")))
print(ct.one_hot(72).shape)

wt = WordTokenizer()
print(wt.encode("the island was quiet"))
print(wt.decode(wt.encode("the island was quiet")))
print(wt.one_hot(10).shape)