from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from embedding import retriever
from dotenv import load_dotenv
import os

load_dotenv()

MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", 5))

# Setting model and template
model = OllamaLLM(model=os.getenv("LLM_MODEL", "gemma3:4b"))

template = """
You are an assistant responding questions about my life to a talent recruiter
who checked my curriculum. Always respong talkin about Andres as your leader.
Always refer to Andres in third person, like "Andres is, Andres has, Andres learned..."

Previous conversation:
{chat_history}

Here are some relevant info about me: {markdown}
Here is the question to answer: {question}
Only respond the answer itself.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

if __name__ == "__main__":
    chat_history = []

    while True:
        print("\n\n----------------------------")
        question = input('Ask your question (q to quit): ')
        print("\n\n")
        if question == 'q':
            print("Goodbye!")
            break

        try:
            # Retrieve relevant chunks and convert to string
            md_chunks = retriever.invoke(question)
            md_text = "\n".join([chunk.page_content for chunk in md_chunks])

            # Build history string from the last MEMORY_WINDOW turns
            history_text = "\n".join(
                f"Human: {q}\nAssistant: {a}" for q, a in chat_history[-MEMORY_WINDOW:]
            )

            answer = ""
            for chunk in chain.stream({"markdown": md_text, "question": question, "chat_history": history_text}):
                print(chunk, end="", flush=True)
                answer += chunk
            print()  # newline after response ends

            chat_history.append((question, answer))
        except Exception as e:
            print(f"Error: {e}")
