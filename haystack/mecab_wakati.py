import MeCab
import sys

def get_wakati_text(jatext:str):
    m = MeCab.Tagger('-Owakati')
    jatext_wakati = m.parse(jatext)
    return jatext_wakati

