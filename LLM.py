from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from embedding import retriever

# Setting model and template
model = OllamaLLM(model='gemma3:4b')

template = """
You are an assistant responding questions about my life to a talent recruiter
who checked my curriculum. 
Here are some relevant info about me: {markdown}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n----------------------------")
    question = input('Ask your question (q to quit): ')
    print("\n\n")
    if question =='q':
        print("Goodbye!")
        break

    try:
        # Retrieve relevant chunks and convert to string
        md_chunks = retriever.invoke(question)
        
        md_text = "\n".join([chunk.page_content for chunk in md_chunks])
        
        for chunk in chain.stream({"markdown": md_text, "question": question}):
            print(chunk, end="", flush=True)
        print()  # newline after response ends
    except Exception as e:
        print(f"Error: {e}")





