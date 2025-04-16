from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
import numpy as np

# Инициализация компонентов Natasha
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def tokenize(sentence):
    doc = Doc(sentence.lower())
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    tokens = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        tokens.append(token.lemma)
    return tokens

def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag