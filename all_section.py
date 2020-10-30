

from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.tokenizer import tokenizer 
from haystack.utils import print_answers 
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever

print("===============DocumentStore=================")
document_store_tfidf = InMemoryDocumentStore()
doc_dir_ja = "data/article_txt_got_ja_0"
dicts_ja = convert_files_to_dicts(dir_path=doc_dir_ja, clean_func=clean_wiki_text, split_paragraphs=True)
print(dicts_ja[0:3])
document_store_tfidf.write_documents(dicts_ja)

print("===============Retriever&Reader================")
retriever_tfidf = TfidfRetriever(document_store=document_store_tfidf)
reader_farm = FARMReader(model_name_or_path="cl-tohoku/bert-base-japanese",use_gpu=True)
finder_tfidf_farm = Finder(reader_farm, retriever_tfidf)


print("===================question========================")
question = "脚本家は誰？"
tokenization = tokenizer.FullTokenizer("./model_sentence_piece/vocab.txt",model_file="./model_sentence_piece/wiki-ja.model",do_lower_case=True)
question_tokenize_bySM = tokenization.space_separation(question)
prediction = finder_tfidf_farm.get_answers(question=question_tokenize_bySM, top_k_retriever=10, top_k_reader=2)

print("\n========answer===========")
print_answers(prediction, details="all")