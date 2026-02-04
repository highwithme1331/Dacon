#Tokenization
import tiktoken

tokenizer = tiktoken.get_encoding("o200k_base")

def tokenize_text(text):
    try:
        token_ids = tokenizer.encode(text)
        tokens = [
            tokenizer.decode([tid]) for tid in token_ids if tokenizer.decode([tid]).strip()
        ]

        return tokens

    except Exception as e:
        print("Error")



#Embedding
import openai

client = openai.OpenAI(api_key='발급 받은 API 키')


words = ['dog', 'lion', 'wolf', 'tiger', 'strawberry', 'pineapple', 'blueberry', 'apple']
groups = ['동물', '동물', '동물', '동물', '과일', '과일', '과일', '과일']

embeddings = []

for word in words:
    response = client.embeddings.create(model="text-embedding-3-small", input=word)
    embedding = response.data[0].embedding
    embeddings.append(embedding)

embeddings = np.array(embeddings)

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

colors = ['red' if g == '동물' else 'blue' for g in groups]

plt.figure(figsize=(8, 6))

for i, (x, y) in enumerate(reduced):
    plt.scatter(x, y, color=colors[i])
    plt.text(x+0.01, y+0.01, words[i], fontsize=10)



#LLM API
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[{"role": "user", "content": "질문 내용"}],
    max_tokens=300,
    temperature=0.1
)



#Setting
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "Negative", 1: "Positive"},
    label2id={"Negative": 0, "Positive": 1}
)



#LoRA
from torch.optim import AdamW 
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    task_type="SEQ_CLS"
)

fine_tuned_model = get_peft_model(model, lora_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuned_model.to(device)
fine_tuned_model.train()


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
optimizer = AdamW(fine_tuned_model.parameters(), lr=5e-5)

for epoch in range(2):
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = fine_tuned_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()