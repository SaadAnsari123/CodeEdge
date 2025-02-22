from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
import gradio as gr

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022",streaming=True)
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", streaming=True)
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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",  # Ensure this is set correctly
)
system_message = "you act like an healthcare assistant"

def stream_response(message, history):
    print(f"Input: {message}. History: {history}\n")

    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_message))

    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    if message is not None:
        history_langchain_format.append(HumanMessage(content=message))
        partial_message = ""
        for response in llm.stream(history_langchain_format):
            partial_message += response.content
            yield partial_message


demo_interface = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
                       container=False,
                       autoscroll=True,
                       scale=7),
)

demo_interface.launch(debug=True, share=True)