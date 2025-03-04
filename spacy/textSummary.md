中文文章的抽取式摘要—使用bert-extractive-summarizer 說明 bert-extractive-summarizer 是一個使用 Bert 加上 Clustering 進行抽取式摘要的模型，詳細原理、實作可以看作者的 Github 有論文連結。因為範例是英文的，用於中文需要稍作修改，載入中文的模型。Github : https://github.com/dmmiller612/bert-extractive-summarizer
安裝需要的套件
pip install bert-extractive-summarizer
pip install spacy==2.3.1
pip install transformers
pip install neuralcoref

python -m spacy download zh_core_web_lg #下載中文的spacy model
載入模型 這裡的 Pretrained Model 可以輸入在 https://huggingface.co 上有的模型名字，或是自己訓練的模型路徑。這裡以 bert-base-chinese 當作範例，可以換成自己常用的模型。載入模型後就可以直接進行摘要了。
# spaCy 載入中文模型
import spacy
import zh_core_web_lg
import neuralcoref
nlp = zh_core_web_lg.load()
neuralcoref.add_to_pipe(nlp)

from summarizer import Summarizer   # summarizer 載入中文模型
from summarizer.sentence_handler import SentenceHandler
from spacy.lang.zh import Chinese
from transformers import *
# Load model, model config and tokenizer via Transformers
modelName = "bert-base-chinese" # 可以換成自己常用的
custom_config = AutoConfig.from_pretrained(modelName)
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained(modelName)
custom_model = AutoModel.from_pretrained(modelName, config=custom_config)
model = Summarizer(
    custom_model=custom_model, 
    custom_tokenizer=custom_tokenizer,
    sentence_handler = SentenceHandler(language=Chinese)
    )
使用模型 模型載入後就可以直接進行文章摘要，model可以傳入一些參數，如果需要可以直接在Github上面找。
body = "要摘要的文章"
result = model(body)
full = ''.join(result)
print(full) # 摘要出來的句子
複製貼上的參數說明
model(
    body: str # The string body that you want to summarize
    ratio: float # The ratio of sentences that you want for the final summary
    min_length: int # Parameter to specify to remove sentences that are less than 40 characters
    max_length: int # Parameter to specify to remove sentences greater than the max length,
    num_sentences: Number of sentences to use. Overrides ratio if supplied.
)
