from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langserve import add_routes
from app.chain import chain as pinecone_wiki_chain

app = FastAPI(
    title="Pinecone Wikipedia RAG API",
    version="1.0",
    description="A RAG API using Pinecone and Wikipedia data",
)

add_routes(app, pinecone_wiki_chain, path="/pinecone-wiki")

@app.get("/")
async def root():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia RAG Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2563eb;
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            font-size: 24px;
            font-weight: 600;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 16px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        .message.user {
            background-color: #2563eb;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            background-color: white;
            color: #1f2937;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .message.loading {
            background-color: #e5e7eb;
            color: #6b7280;
        }
        #input-container {
            padding: 20px;
            background-color: white;
            border-top: 1px solid #e5e7eb;
            display: flex;
            gap: 12px;
        }
        #question {
            flex: 1;
            padding: 12px 16px;
            font-size: 16px;
            border: 1px solid #d1d5db;
            border-radius: 24px;
            outline: none;
            transition: border-color 0.2s;
        }
        #question:focus {
            border-color: #2563eb;
        }
        #submit {
            padding: 12px 24px;
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 24px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        #submit:hover:not(:disabled) {
            background-color: #1d4ed8;
        }
        #submit:disabled {
            background-color: #9ca3af;
            cursor: not-allowed;
        }
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background-color: #9ca3af;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌍 Aqib's RAG Assistant</h1>
    </div>
    <div id="chat-container"></div>
    <div id="input-container">
        <input type="text" id="question" placeholder="Ask me anything about Wikipedia content..." autocomplete="off">
        <button id="submit">Send</button>
    </div>
    <script>
        const chatContainer = document.getElementById('chat-container');
        const questionInput = document.getElementById('question');
        const submitButton = document.getElementById('submit');
        
        let conversationHistory = [];

        function addMessage(text, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'assistant');
            
            if (text === 'typing') {
                messageDiv.classList.add('loading');
                messageDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
            } else {
                messageDiv.textContent = text;
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return messageDiv;
        }

        async function sendQuestion() {
            const originalQuestion = questionInput.value.trim();
            if (!originalQuestion) return;

            addMessage(originalQuestion, true);
            questionInput.value = '';
            submitButton.disabled = true;

            const loadingMsg = addMessage('typing', false);

            // Build contextual question with conversation history
            let questionWithContext = originalQuestion;
            if (conversationHistory.length > 0) {
                // Format conversation history more clearly
                const recent = conversationHistory.slice(-4); // Last 2 exchanges
                const context = recent.map((msg, idx) => {
                    if (idx % 2 === 0) {
                        return 'Q: ' + msg.content;
                    } else {
                        return 'A: ' + msg.content;
                    }
                }).join('\\n');
                
                questionWithContext = '[Conversation so far]\\n' + context + '\\n\\n[New question]: ' + originalQuestion;
            }

            try {
                const response = await fetch('/pinecone-wiki/invoke', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ input: questionWithContext })
                });

                const data = await response.json();
                loadingMsg.remove();
                addMessage(data.output, false);
                
                // Store in history
                conversationHistory.push({ role: 'User', content: originalQuestion });
                conversationHistory.push({ role: 'Assistant', content: data.output });
                
                // Keep last 8 messages (4 exchanges)
                if (conversationHistory.length > 8) {
                    conversationHistory = conversationHistory.slice(-8);
                }
                
            } catch (error) {
                loadingMsg.remove();
                addMessage('Error: ' + error.message, false);
            }

            submitButton.disabled = false;
            questionInput.focus();
        }

        submitButton.addEventListener('click', sendQuestion);
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendQuestion();
        });

        questionInput.focus();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)