import Mecab
import sys

mecab = Mecab.Tagger('-Owakati')

input_text = []
if len(sys.argv)<2:
    try:
        while True:
            input_text.append(input())
    except EOFError:
        pass

else:
    input_text.append(sys.argv[1])

for text in input_text:
    output=mecab.parse(text)
    print(output,end="")
