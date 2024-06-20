import torch
from transformers import BertModel
from transformers import BertTokenizer

proxies = {'http': '127.0.0.1:7890', 'https': '127.0.0.1:7890'}
# 加载预训练的BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', proxies=proxies)

# 文本输入
text = "她是中国人"

tokens = tokenizer.tokenize(text, add_special_tokens=True)
# 使用BERT tokenizer进行token化
encoded_input = tokenizer.encode(text, add_special_tokens=True)

print("Tokens: ", tokens)
print("Tokenized IDs: ", encoded_input)

# 加载预训练的BERT模型
model = BertModel.from_pretrained('bert-base-chinese', proxies=proxies)

# 转换为模型需要的输入格式
inputs = torch.tensor([encoded_input])

# 获取嵌入表示
with torch.no_grad():
    outputs = model(inputs)
    embeddings = outputs.last_hidden_state

print("Embeddings shape: ", embeddings.shape)
