#Basic Data Cleaning
lowercase_text = text.lower()
cleaned_text = lowercase_text.strip()

cleaned_text = text.replace('#', '').replace('@', '')



#Regular Expression Data Cleaning
import re

pattern = r'\s+'
cleaned_text = re.sub(pattern, ' ', text)

pattern = r'\d+'
cleaned_text = re.sub(pattern, '', text)



#Emoji Data Cleaning
import emoji

cleaned_text = emoji.replace_emoji(text, replace='')



#HTML Data Cleaning
from bs4 import BeautifulSoup

soup = BeautifulSoup(html_doc, 'html.parser')
cleaned_text = soup.get_text()



#Whitespace Tokenizer
def tokenizer(text):
    tokens = text.split()
    
    return tokens

tokens = tokenizer(text)



#Regular Expression Tokenizer
import re

def tokenizer(text):
    pattern = r'\b\w+\b'
    tokens = re.findall(pattern, text)
    
    return tokens

tokens = tokenizer(text)



#BPE Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=10000, min_frequency=10)
tokenizer.train(train_files, trainer)
output = tokenizer.encode(text)



#WordPiece Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(vocab_size=20000, min_frequency=2)
tokenizer.train(train_files, trainer)
output = tokenizer.encode(text)



#GPT Tokenizer
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
encoded_input = tokenizer.encode(text)
tokens = tokenizer.convert_ids_to_tokens(encoded_input)



#Word2Vec
from gensim.models import Word2Vec

model_wv = Word2Vec(sentences, vector_size=5, window=5, min_count=1, workers=4)
wv_matrix = torch.FloatTensor([model_wv.wv[word] for word in model_wv.wv.index_to_key])
w2v_embedding_layer = nn.Embedding.from_pretrained(wv_matrix, freeze=False)
input_data = torch.LongTensor([0, 1])
embedded_data = w2v_embedding_layer(input_data)



#Pre-Training Word2Vec
wv_vectors = api.load('word2vec-ruscorpora-300')
weights = torch.FloatTensor(wv_vectors.vectors)
wv_embedding_layer = nn.from_pretrained(weights, freeze=False)
input_data = torch.LongTensor([0, 1])
embedded_data = wv_embedding_layer(input_data)



#Pre-Training GloVe
import gensim.downloader as api

glove_vectors = api.load('glove-wiki-gigaword-50')
weights = torch.FloatTensor(glove_vectors.vectors)
glove_embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False)
input_data = torch.LongTensor([0, 1])
embedded_data = glove_embedding_layer(input_data)



#Pre-Training FastText
word_vectors = api.load('fasttext-wiki-news-subwords-300’)
weights = torch.FloatTensor(word_vectors.vectors)
embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False)
input_data = torch.LongTensor([0, 1])
embedded_data = embedding_layer(input_data)