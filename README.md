# Voice Agent RAG — AI Real-Estate Voice Assistant
An **AI-powered real-estate voice assistant** for the Cairo / New Cairo market. Users speak in **Egyptian Arabic**; the system transcribes speech, extracts lead details with an LLM, searches thousands of listings via **RAG** (ChromaDB), presents matches conversationally, and sends qualified leads to **n8n** for sales follow-up.


Python · FastAPI · WebSockets · RAG · ChromaDB · Gemini · Deepgram · Edge TTS · n8n

Built a voice-first sales assistant for Cairo real estate that lets users search listings and submit leads in Egyptian Arabic. The system streams microphone audio over WebSockets, transcribes speech with Deepgram, retrieves properties from a ChromaDB vector index, and responds with neural TTS while pushing qualified leads to n8n.

Designed an end-to-end voice pipeline (STT → LLM field extraction → RAG search → TTS) with a FastAPI WebSocket server and a linear sales-funnel state machine for lead qualification (name, budget, location, property choice, email).
Implemented RAG over 11,000+ property listings using HuggingFace embeddings and ChromaDB, combining semantic search with metadata filters for budget and location to return relevant Cairo/New Cairo matches.
Integrated Google Gemini for structured JSON extraction and intent classification, Microsoft Edge TTS (ar-EG-SalmaNeural) for Arabic speech output, and n8n webhooks to automate CRM lead handoff after each completed session.
 


