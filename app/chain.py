import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)

NEW_TOPIC_STARTERS = [
    "what is", "what are", "who is", "who was", "who were",
    "tell me about", "explain", "describe", "define",
    "what do you know about", "search for", "look up", "find",
    "i want to know about", "can you explain", "give me information",
]

# These are meta/formatting instructions — never search Pinecone for these
META_INSTRUCTIONS = [
    "too short", "too long", "too brief", "too detailed", "too much",
    "more detail", "more details", "more depth", "simpler", "simplify",
    "shorter", "longer", "expand", "elaborate", "summarize", "summary",
    "don't give", "give me", "make it", "be more", "be less",
    "that's too", "thats too", "not enough", "enough", "perfect",
    "good", "ok", "okay", "got it", "i see", "interesting",
    "thanks", "thank you", "great", "nice", "cool", "awesome",
    "continue", "go on", "keep going", "next", "and then", "what else",
    "anything else", "tell me more", "more info", "more information",
]


def is_new_topic(question: str, has_cached_context: bool) -> bool:
    """
    Returns True only if the user is clearly asking about a brand new topic.
    Meta instructions and follow-ups always reuse cached context.
    """
    # No cache yet — must search
    if not has_cached_context:
        return True

    q = question.lower().strip()

    # Check if it's a meta/formatting instruction — never search for these
    for meta in META_INSTRUCTIONS:
        if meta in q:
            print(f"💬 Meta instruction detected: '{question}' — reusing context")
            return False

    # Very short messages (≤5 words) with no clear topic = follow-up
    words = q.split()
    if len(words) <= 5:
        return False

    # Starts with a new-topic phrase → new search
    for starter in NEW_TOPIC_STARTERS:
        if q.startswith(starter):
            return True

    # Long messages that don't start with a follow-up word → new topic
    if len(words) > 10:
        followup_starters = ["why", "how", "when", "where", "who", "what",
                             "can", "could", "would", "is", "are", "was",
                             "were", "do", "does", "did", "tell", "give",
                             "show", "explain", "elaborate", "more", "also"]
        first_word = words[0] if words else ""
        if first_word not in followup_starters:
            return True

    return False


def fetch_and_answer(question: str, history: list[dict] | None = None,
                     cached_context: str | None = None) -> tuple[str, str]:
    """
    Answers using BOTH Wikipedia context AND the LLM's own training knowledge.
    Meta instructions (too short, too long, etc.) reuse cached context.
    """
    if history is None:
        history = []

    print(f"\n🔍 Question: {question}")

    # ── Decide: new search or reuse cached context ─────────────────────────
    if is_new_topic(question, bool(cached_context)):
        print("🔎 New topic — searching Pinecone...")
        docs = vectorstore.similarity_search(question, k=5)

        if docs:
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            print(f"📊 Retrieved {len(docs)} chunks ({len(context):,} chars)")
        else:
            context = ""
            print("⚠️  No Pinecone results — using LLM knowledge only")
    else:
        context = cached_context or ""
        print("♻️  Reusing cached context")

    # ── System prompt ──────────────────────────────────────────────────────
    system_prompt = """You are a highly knowledgeable assistant with the communication style of Claude by Anthropic — clear, structured, warm, and genuinely helpful.

Answer using both the Wikipedia context provided AND your own broad training knowledge combined.

FORMATTING (follow strictly):
- Use **bold** for key terms, important concepts, and section headers
- Use bullet points for multiple items, steps, features, or comparisons
- Break answers into clearly **bold-labeled sections** when covering multiple aspects
- For definitions: 1 crisp sentence first, then expand
- For explanations: use analogies and real-world examples
- Highlight the most important takeaway when relevant

LENGTH:
- Simple questions → 3-5 sentences, no headers needed
- Medium questions → 2-3 sections with bullets
- Complex questions → full structured response with bold headers and bullets
- "too short" → significantly expand the SAME topic with more sections
- "too long" → condense the SAME topic, keep only core ideas
- "elaborate" or "more" → add deeper explanation on the SAME topic
- Never pad with filler — every sentence must earn its place

QUALITY:
- Open with the most direct useful answer first — no preamble
- Support with structure, examples, and context
- Use analogies to make complex ideas feel intuitive
- Be conversational and confident, not robotic
- Prefer Wikipedia context for specific facts when available
- Never say "based on the context" or "according to Wikipedia" — just answer naturally
- If you don't know something, say so briefly"""

    # ── Build messages ─────────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    if context:
        user_content = (
            f"Here is some relevant Wikipedia information on this topic:\n\n"
            f"{context}\n\n"
            f"Question: {question}"
        )
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})

    answer = llm.invoke(messages).content
    return answer, context


chain = fetch_and_answer