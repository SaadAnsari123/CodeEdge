import os
import pandas as pd
import numpy as np
import gradio as gr
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

precautions_dict = {
    "Fungal infection": ["Keep affected area clean and dry", "Use antifungal creams", "Avoid sharing personal items", "Wear loose-fitting clothes"],
    "Allergy": ["Avoid allergens", "Take antihistamines", "Use air purifiers", "Keep surroundings clean"],
    "GERD": ["Avoid spicy food", "Eat smaller meals", "Do not lie down immediately after eating", "Maintain healthy weight"],
    "Chronic cholestasis": ["Follow a low-fat diet", "Stay hydrated", "Take prescribed medications", "Avoid alcohol"],
    "Drug Reaction": ["Stop medication causing reaction", "Consult a doctor immediately", "Take antihistamines", "Drink plenty of water"],
    "Peptic ulcer disease": ["Avoid spicy food", "Eat a balanced diet", "Reduce stress", "Take prescribed medications"],
    "AIDS": ["Take antiretroviral therapy", "Practice safe sex", "Maintain good hygiene", "Eat a healthy diet"],
    "Diabetes": ["Monitor blood sugar levels", "Follow a balanced diet", "Exercise regularly", "Take prescribed medications"],
    "Gastroenteritis": ["Drink plenty of fluids", "Eat light meals", "Avoid dairy products", "Maintain hygiene"],
    "Bronchial Asthma": ["Avoid allergens", "Use inhalers as prescribed", "Stay away from smoke", "Practice breathing exercises"],
    "Hypertension": ["Reduce salt intake", "Exercise regularly", "Maintain a healthy weight", "Monitor blood pressure"],
    "Migraine": ["Avoid bright lights and loud noises", "Stay hydrated", "Manage stress", "Get enough sleep"],
    "Cervical spondylosis": ["Maintain proper posture", "Do neck exercises", "Use a supportive pillow", "Apply hot or cold compress"],
    "Paralysis (brain hemorrhage)": ["Follow physiotherapy", "Take prescribed medications", "Eat a healthy diet", "Monitor blood pressure"],
    "Jaundice": ["Stay hydrated", "Eat a healthy diet", "Avoid alcohol", "Get adequate rest"],
    "Malaria": ["Use mosquito repellents", "Sleep under mosquito nets", "Take antimalarial medications", "Wear full-sleeved clothes"],
    "Chicken pox": ["Avoid scratching", "Take antiviral medications", "Use calamine lotion", "Drink plenty of fluids"],
    "Dengue": ["Use mosquito repellents", "Stay hydrated", "Wear protective clothing", "Rest well"],
    "Typhoid": ["Drink purified water", "Maintain hygiene", "Take prescribed antibiotics", "Avoid raw foods"],
    "Hepatitis A": ["Avoid alcohol", "Eat a balanced diet", "Get vaccinated", "Maintain hygiene"],
    "Hepatitis B": ["Avoid alcohol", "Get vaccinated", "Practice safe sex", "Eat a healthy diet"],
    "Hepatitis C": ["Avoid alcohol", "Take antiviral therapy", "Eat a balanced diet", "Avoid sharing needles"],
    "Hepatitis D": ["Avoid alcohol", "Get vaccinated for Hepatitis B", "Maintain good hygiene", "Eat a healthy diet"],
    "Hepatitis E": ["Drink clean water", "Eat hygienic food", "Avoid alcohol", "Take proper rest"],
    "Alcoholic hepatitis": ["Stop alcohol consumption", "Eat a nutritious diet", "Stay hydrated", "Follow doctor’s advice"],
    "Tuberculosis": ["Take prescribed medication", "Cover mouth while coughing", "Maintain good ventilation", "Eat a healthy diet"],
    "Common Cold": ["Stay hydrated", "Rest properly", "Take steam inhalation", "Avoid cold weather"],
    "Pneumonia": ["Take prescribed antibiotics", "Stay hydrated", "Get vaccinated", "Rest properly"],
    "Dimorphic hemorrhoids (piles)": ["Eat a high-fiber diet", "Drink plenty of water", "Avoid straining during bowel movements", "Exercise regularly"],
    "Heart attack": ["Follow a heart-healthy diet", "Avoid smoking", "Exercise regularly", "Manage stress"],
    "Varicose veins": ["Avoid standing for long periods", "Exercise regularly", "Wear compression stockings", "Elevate legs while resting"],
    "Hypothyroidism": ["Take prescribed thyroid medication", "Eat iodine-rich foods", "Exercise regularly", "Avoid stress"],
    "Hyperthyroidism": ["Take prescribed medications", "Avoid caffeine", "Eat a balanced diet", "Manage stress"],
    "Hypoglycemia": ["Eat frequent small meals", "Avoid skipping meals", "Monitor blood sugar levels", "Carry glucose tablets"],
    "Osteoarthritis": ["Do low-impact exercises", "Maintain a healthy weight", "Use hot/cold therapy", "Take prescribed pain relievers"],
    "Arthritis": ["Exercise regularly", "Maintain a healthy weight", "Use assistive devices if needed", "Take prescribed medications"],
    "(Vertigo) Paroxysmal positional vertigo": ["Avoid sudden head movements", "Do balance exercises", "Stay hydrated", "Take prescribed medication"],
    "Acne": ["Wash face twice daily", "Avoid oily foods", "Use non-comedogenic products", "Avoid touching the face"],
    "Urinary tract infection": ["Drink plenty of water", "Maintain hygiene", "Avoid holding urine", "Take prescribed antibiotics"],
    "Psoriasis": ["Moisturize skin regularly", "Avoid triggers like stress", "Take prescribed medications", "Use medicated shampoos"],
    "Impetigo": ["Maintain good hygiene", "Avoid scratching", "Use antibiotic ointments", "Wash affected area regularly"],
    "Prognosis": ["Follow doctor’s advice", "Take prescribed medications", "Maintain a healthy lifestyle", "Get regular check-ups"]
}

# Initialize chatbot LLM
llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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" # Set your OpenRouter API key in environment variables
)

system_message = "You act like a healthcare assistant."

def stream_response(message, history):
    history_langchain_format = [SystemMessage(content=system_message)]
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    if message:
        history_langchain_format.append(HumanMessage(content=message))
        partial_message = ""
        for response in llm.stream(history_langchain_format):
            partial_message += response.content
            yield partial_message

# Load dataset
df = pd.read_csv('Training.csv')
y = df.iloc[:, -1]   # Target (Disease)
X = pd.get_dummies(df.iloc[:, :-1])  # Features (Symptoms)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

def predict_disease(symptoms):
    symptom_list = [sym.strip().lower() for sym in symptoms.split(",")]
    user_symptoms = {col: (1 if col in symptom_list else 0) for col in X.columns}
    input_data = pd.DataFrame([user_symptoms], columns=X.columns)
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data).max() * 100
    precautions = precautions_dict.get(prediction, ["No specific precautions found"])
    precautions_text = "\n".join([f"✅ {p}" for p in precautions])
    return f"Predicted Disease: {prediction}\nChance of Occurrence: {prediction_proba:.2f}\nPrecautions: {precautions_text}%"



with gr.Blocks() as interface1:
    gr.Markdown("### 🩺 AI Disease Predictor & Chatbot")
    with gr.Row():
        symptoms_input = gr.Textbox(label="Enter Symptoms", placeholder="e.g., fever, cough, headache")
        predict_button = gr.Button("Predict Disease")
        output_text = gr.Textbox(label="Prediction Result")
    predict_button.click(predict_disease, inputs=symptoms_input, outputs=output_text)
    a = gr.ChatInterface(fn=stream_response, title="Healthcare Assistant Chatbot")
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


# demo_interface = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
#                        container=False,
#                        autoscroll=True,
#                        scale=7),
# )

# demo_interface.launch(debug=True, share=True)



interface1.launch()