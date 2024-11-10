from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def calculate_rouge(reference, generated):
    rouge = Rouge()
    return rouge.get_scores(generated, reference, avg=True)

def calculate_bleu(reference, generated):
    return sentence_bleu([reference.split()], generated.split())
