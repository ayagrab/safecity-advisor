let questionAsked = false;

async function sendMessage() {
    if (questionAsked) return;

    const input = document.getElementById("user-input");
    const answerBox = document.getElementById("answer-box");
    const sendBtn = document.getElementById("send-btn");

    const message = input.value.trim();
    if (!message) return;

    questionAsked = true;
    input.disabled = true;
    sendBtn.disabled = true;

    answerBox.innerHTML = `
        <div class="qa-card user-card fade-in">
            <h4>👤 Your Question</h4>
            <p>${escapeHtml(message)}</p>
        </div>

        <div class="qa-card bot-card fade-in">
            <h4>🤖 SafeCity Advisor</h4>
            <div class="typing">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: message})
        });

        const data = await response.json();

        answerBox.innerHTML = `
            <div class="qa-card user-card fade-in">
                <h4>👤 Your Question</h4>
                <p>${escapeHtml(message)}</p>
            </div>

            <div class="qa-card bot-card fade-in">
                <h4>🤖 SafeCity Advisor</h4>
                <p>${escapeHtml(data.answer)}</p>
            </div>
        `;
    } catch (error) {
        answerBox.innerHTML = `
            <div class="qa-card bot-card fade-in">
                <h4>⚠️ Error</h4>
                <p>Something went wrong. Please restart and try again.</p>
            </div>
        `;
    }
}

function restartChat() {
    questionAsked = false;

    const input = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const answerBox = document.getElementById("answer-box");

    input.disabled = false;
    sendBtn.disabled = false;
    input.value = "";
    input.focus();

    answerBox.innerHTML = `
        <div class="welcome fade-in">
            <h2>Hello, I’m SafeCity Advisor.</h2>
            <p>Ask one question about smart-city technologies, surveillance, safety, privacy, or oversight.</p>
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

document.getElementById("user-input").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});