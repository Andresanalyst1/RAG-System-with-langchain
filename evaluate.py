from dotenv import load_dotenv
import os
import json

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from embedding import retriever
from LLM import chain

LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:335m")

# Judge LLM is Claude (same model used for generation). Embeddings stay local
# via Ollama since RAGAS only uses them for cosine similarity in AnswerRelevancy.
ragas_llm = LangchainLLMWrapper(
    ChatAnthropic(
        model=LLM_MODEL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
)
ragas_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBEDDING_MODEL))

# Load test questions and ground truths
with open("eval_dataset.json") as f:
    eval_data = json.load(f)

print(f"\n[INFO] Running evaluation on {len(eval_data)} questions...\n")

rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for item in eval_data:
    question = item["question"]
    ground_truth = item["ground_truth"]

    chunks = retriever.invoke(question)
    contexts = [c.page_content for c in chunks]
    md_text = "\n".join(contexts)

    answer = chain.invoke({"markdown": md_text, "question": question, "chat_history": ""})

    rows["question"].append(question)
    rows["answer"].append(answer)
    rows["contexts"].append(contexts)
    rows["ground_truth"].append(ground_truth)

    print(f"  Q: {question}")
    print(f"  A: {answer[:120]}{'...' if len(answer) > 120 else ''}\n")

dataset = Dataset.from_dict(rows)

results = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],
    llm=ragas_llm,
    embeddings=ragas_emb,
)

print("\n====== RAGAS Evaluation Results ======")
df = results.to_pandas()
df.insert(0, "question", rows["question"])
print(df[["question", "faithfulness", "answer_relevancy"]].to_string(index=False))
print(f"\nMean faithfulness:      {df['faithfulness'].mean():.3f}")
print(f"Mean answer_relevancy:  {df['answer_relevancy'].mean():.3f}")
