# from gramformer import Gramformer
import dill
import nltk
from nltk.tokenize import sent_tokenize

with open('gf.pth', 'rb') as f:
    gf_inference = dill.load(f) #load model from .pth file
nltk.download('punkt')

text = input("Enter essay: ") # get input

output_edits = []
output_result = []
sentences = sent_tokenize(text, language='english') # split a essay to many sentences

for sentence in sentences:
    result = gf_inference.correct(sentence, max_candidates=1) # correct sentence
    wrong_word = gf_inference.get_edits(sentence, list(result)[0]) # get wrong edits
    output_edits.extend(wrong_word)
    output_result.append(list(result)[0])

print([edit for edit in output_edits]) # list of wrong edits
print(" ".join(output_result)) # the essay after correction