# 🚚 AI-Driven Logistics Route Builder

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)

An intelligent Telegram Bot backend designed to automate and optimize logistics routing. This system allows freight carriers and delivery drivers to simply send screenshots of delivery addresses, which the bot processes using Optical Character Recognition (OCR) and Generative AI to return an optimized daily route manifest.

## 🏗️ System Architecture & Workflow

The backend is built with **FastAPI** to handle asynchronous requests and Webhook integrations efficiently. The data flow is as follows:
1. **User Interaction:** The driver interacts with the Telegram bot (Start new route, delete current, add screenshots).
2. **Data Extraction:** Uploaded images are processed via **Google Cloud Vision API (OCR)** to extract raw text.
3. **AI Processing:** The raw text is passed to **Gemini API / OpenAI API** to identify, normalize, and structure the delivery addresses.
4. **Resilience & Fallback:** If the AI model times out or the backend encounters an error, the system safely catches the exception and provides the user with an interactive "Retry" option, ensuring high availability.

## ✨ Key Features
* **Interactive UI via Telegram:** Menu-driven workflow for end-users (drivers).
* **Multi-AI Integration:** Combines Google Vision for OCR with LLMs (OpenAI/Gemini) for natural language parsing.
* **Fault Tolerance:** Built-in retry mechanisms for third-party API failures.

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* Telegram Bot Token (from BotFather)
* API Keys for OpenAI, Gemini, and Google Cloud Vision.

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/WmikeeD/ai-route-builder.git](https://github.com/WmikeeD/ai-route-builder.git)
   cd ai-route-builder
Install dependencies:

Bash
pip install -r requirements.txt
Configure Environment Variables (.env):

Fragmento de código
TELEGRAM_TOKEN=your_telegram_token
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
PORT=8000
Run the application:

Bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
