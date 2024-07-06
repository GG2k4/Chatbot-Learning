from transformers import pipeline
import gradio as gr

chatbot = pipeline(model="facebook/blenderbot-400M-distill")

message_list = []
response_list = []

def vanilla_chatbot(message, history):
    global message_list, response_list
    message_list.extend(history)
    message_list.append(message)
    response = chatbot(message)
    response_list.append(response[0]['generated_text'])
    return response[0]['generated_text']

demo_chatbot = gr.ChatInterface(vanilla_chatbot, title="Vanilla Chatbot", description="Enter text to start chatting.")

demo_chatbot.launch()