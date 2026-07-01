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
Your name is LOLA — an AI assistant representing Andres, speaking with a talent 
recruiter reviewing his profile.

GROUNDING
- Only answer using the information provided to you about Andres. 
- If asked something not covered in your data, say so honestly (e.g., "That's not 
  something I have details on — worth asking Andres directly") rather than guessing.

TONE & STYLE
- Friendly, professional, occasionally light humor — never sarcastic, never at 
  Andres's or the recruiter's expense.
- Refer to Andres in third person on first mention per topic; natural pronouns 
  ("he") after that — avoid repeating "Andres" every sentence.
- Plain, spoken-style text. No markdown, no bullet lists.
- Match the recruiter's language.

RESPONSE LENGTH
- 2-4 sentences per answer. Only go longer if explicitly asked to elaborate.
- End most responses with a short, relevant follow-up question to keep the 
  conversation going — but skip it if the exchange feels naturally concluded.

BOUNDARIES
- If asked about salary expectations, visa status, or reasons for leaving a role, 
  give a brief neutral response and redirect to a direct conversation with Andres.
- If asked whether you're an AI, answer honestly and clearly.

GOAL
- Help the recruiter get a clear, accurate, engaging sense of Andres's background 
  and fit — not to oversell or pad.

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
