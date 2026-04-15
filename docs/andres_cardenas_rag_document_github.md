# Andres Cardenas — RAG Knowledge Document
**Data Scientist · ML Engineer · Data Analyst**  
Brisbane, Queensland, Australia  
*Generated as an AI assistant knowledge base source — April 2026*

---

## About

Andres Cardenas is an Industrial Engineer who transitioned into Data Science and Machine Learning, with approximately two years of focused study and hands-on project development. He is currently based in Brisbane, Australia, and is actively seeking roles in data analytics, data science, or machine learning engineering.

Andres is passionate about applying AI to real-world problems, particularly around privacy-respecting local AI infrastructure and end-to-end ML deployments. He believes in building practical, deployable systems rather than just theoretical notebooks.

Outside of work, Andres enjoys weightlifting, running, and cooking. He tracks and analyses his running training using Strava and is building AI-powered tools to improve his athletic performance.

---

## Contact & Links

- **GitHub:** github.com/Andresanalyst1
- **LinkedIn:** linkedin.com/in/andres-cardenas-4b992a191
- **Email:** andrescardenas653@gmail.com
- **Location:** Brisbane, Queensland, Australia

---

## Technical Skills

| Category | Tools & Technologies |
|---|---|
| Languages | Python, SQL (MySQL) |
| ML & Data Science | scikit-learn, CatBoost, NumPy, Pandas, Matplotlib |
| AI / LLM Stack | LangChain, ChromaDB, Ollama, RAG pipelines, Gemini Flash API, Anthropic Claude API |
| Deployment & Apps | Streamlit, Streamlit Community Cloud, Docker (WSL/Ubuntu) |
| Infrastructure | Open WebUI, Local LLM deployment, Raspberry Pi CM4, Nginx |
| Cloud & Other | AWS, Git, GitHub, Power BI |
| Agent Frameworks | OpenClaw, Discord bot integration, Strava API, ElevenLabs TTS, Microsoft Edge TTS |

---

## Projects

### RAG-Based AI Chatbot for Recruiters
**Stack:** Python · LangChain · ChromaDB · Ollama · Gemini Flash · Streamlit · RAG  
**Link:** github.com/Andresanalyst1

An end-to-end RAG chatbot designed to answer recruiter questions about Andres's profile. Accessible via QR code on his CV. The local prototype uses Ollama + LangChain + ChromaDB. The cloud version uses Gemini Flash API + Streamlit Community Cloud for 24/7 availability. Features pre-computed embeddings committed to GitHub and API key management via `st.secrets`.

---

### ML Australian Car Market — Price Predictor
**Stack:** Python · CatBoost · scikit-learn · Pandas · Streamlit · Streamlit Cloud  
**Link:** github.com/Andresanalyst1/ML-Australian-Car-Market

End-to-end machine learning project to predict vehicle prices in the Australian market. Uses a CatBoost regression model with hyperparameter tuning via RandomizedSearchCV. Includes exploratory data analysis, feature engineering, model serialisation with joblib/pickle, and a deployed Streamlit web app on Streamlit Community Cloud.

---

### Australia Car Market — Seller Price Tool
**Stack:** Python · Pandas · Jupyter Notebook · Data Analysis  
**Link:** github.com/Andresanalyst1/Australia_Car_Market

A companion project helping potential sellers find the optimal listing price for their vehicles based on Australian market data. Includes data analysis with Jupyter Notebooks and visualisation.

---

### Local LLM + RAG Company Infrastructure (Internal)
**Stack:** Open WebUI · Ollama · Docker · Nginx · RAG · LLM Infrastructure

Designed and advocated for a company-wide local LLM and RAG chatbot system motivated by data privacy compliance. Stack includes Open WebUI + Ollama on a Linux VM with GPU, HTTPS/Nginx security layer, and a document assistant powered by the RAG/Knowledge Base features of Open WebUI. Drafted IT infrastructure request documents for internal approval. This project reflects Andres's understanding of the compliance risks of routing sensitive company data through external APIs.

---

### LOLA — Personal AI Agent on Raspberry Pi CM4 *(Active Development)*
**Stack:** Raspberry Pi CM4 · OpenClaw · Anthropic Claude API · Discord · Microsoft Edge TTS · ARM64  
**Status:** Actively developed and self-hosted

LOLA is Andres's personal AI agent running on a Raspberry Pi CM4 (eMMC storage, ARM64 architecture). It uses the OpenClaw agent framework with the Anthropic Claude API as the LLM backend, and is interfaced primarily through a Discord bot with Microsoft Edge TTS for voice output. The gateway runs on port 18789 on the local network (IP: 192.168.20.19).

This project involved extensive end-to-end hardware and software work: flashing eMMC via rpiboot (USB-C port on the 52Pi EP-0146 board), bootloader and jumper configuration, PATH setup for the OpenClaw binary, session management debugging, and TTS provider evaluation (tested ElevenLabs, OpenAI TTS, and Microsoft Edge TTS before settling on Edge TTS for stability). LOLA demonstrates Andres's ability to architect, deploy, and maintain a full self-hosted AI stack from bare metal upward.

---

### LOLA × Strava — AI Running Coach *(Active Development)*
**Stack:** OpenClaw · Anthropic Claude API · Strava API · Raspberry Pi CM4 · Python  
**Status:** In development as a hobby/personal project

Andres is integrating his OpenClaw agent LOLA with the Strava API to create a personalised AI running coach. The integration allows LOLA to read Andres's training data from Strava — including run distances, pace, heart rate, elevation, and workout history — and provide coaching feedback, training load analysis, and improvement suggestions.

This project sits at the intersection of Andres's technical skills (API integration, agent tool development, data analysis) and his personal passion for running. It demonstrates applied LLM agent development with real-world fitness data and reflects his broader interest in building AI tools that improve everyday life.

---

## Publications

**Australian Journal of Educational Technology (AJET)**  
Andres is the author of a peer-reviewed scientific article published in the Australian Journal of Educational Technology.  
Reference: ajet.org.au/index.php/AJET/article/view/6512

---

## Education & Continuous Learning

- **Industrial Engineering degree** — formal academic background providing strong analytical and systems-thinking foundations.
- **Data Science & Machine Learning** — approximately 2 years of structured self-study via Codecademy.
- Hands-on project-based learning across the full ML lifecycle: data wrangling, EDA, modelling, evaluation, and cloud deployment.
- Active practitioner in LLM/RAG infrastructure, local AI deployment, and AI agent frameworks.
- Continuous learner who documents and publicly shares work via GitHub.

---

## Professional Profile & Working Style

Andres is a methodical, hands-on professional who approaches problems by understanding the underlying "why" before implementing solutions. He is comfortable debugging complex multi-layer systems spanning hardware, OS, networking, and application stack, and iterates systematically until resolution. He communicates fluently in both English and Spanish.

- **Debugging & problem-solving mindset** — root-cause analysis, systematic iteration, comfortable with low-level hardware and high-level ML abstraction simultaneously.
- **Full-stack ML awareness** — from raw data and model training through to cloud deployment and user-facing interfaces.
- **Privacy-first thinking** — motivated to build AI systems that keep sensitive data local and secure; understands compliance risk in enterprise AI contexts.
- **Entrepreneurial portfolio approach** — builds and deploys real products (not just notebooks) to demonstrate skills to recruiters; uses QR codes on his CV to link live demos.
- **Collaborative** — experience working with stakeholders to draft internal AI infrastructure proposals and business cases.
- **Self-hosted AI enthusiast** — runs personal AI infrastructure at home, including a local agent on a Raspberry Pi and Open WebUI with Ollama for private LLM access.

---

## Personal Interests

- **Weightlifting** — regular gym training.
- **Ukulele** - Learning how to play ukulele replicating his favourites songs.
- **Running** — tracks training on Strava; currently building an AI coaching tool on top of his running data.
- **Cooking** — enjoys preparing meals.
- **Self-hosted technology** — passionate about owning and controlling personal AI infrastructure.

---

*This document is intended as a RAG knowledge base source for an AI assistant representing Andres Cardenas. It should be chunked by section (About, Projects, Skills, etc.) for optimal retrieval performance.*
