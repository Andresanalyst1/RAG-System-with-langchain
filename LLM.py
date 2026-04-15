from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from embedding import retriever
from dotenv import load_dotenv
import os

load_dotenv()

MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", 5))

# Setting model and template
model = ChatAnthropic(
    model=os.getenv("LLM_MODEL"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
)


template = """
Your name is LOLA. You are an assistant responding questions about Andres life to a talent recruiter
who is checking his curriculum.
Be friendly and respond open questions, always interacting with the Andres info.
Always refer to Andres in third person, like "Andres is, Andres has, Andres learned..."
Be concise in your answers. Respond with short statements and leave open questions at
the end of your respond to keep the conversation open.

Previous conversation:
{chat_history}

Here are some relevant info about me: {markdown}
Here is the question to answer: {question}
Only respond the answer itself.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model | StrOutputParser()


def stream_answer(question: str, chat_history: list[tuple[str, str]]):
    """Yield response tokens for `question`, using the last MEMORY_WINDOW turns of history."""
    md_chunks = retriever.invoke(question)
    md_text = "\n".join(chunk.page_content for chunk in md_chunks)
    history_text = "\n".join(
        f"Human: {q}\nAssistant: {a}" for q, a in chat_history[-MEMORY_WINDOW:]
    )
    yield from chain.stream({
        "markdown": md_text,
        "question": question,
        "chat_history": history_text,
    })


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
            answer = ""
            for chunk in stream_answer(question, chat_history):
                print(chunk, end="", flush=True)
                answer += chunk
            print()  # newline after response ends

            chat_history.append((question, answer))
        except Exception as e:
            print(f"Error: {e}")
