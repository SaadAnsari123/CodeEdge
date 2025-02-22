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
import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
doctor_map = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist",
    "Diabetes": "Endocrinologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedic",
    "Parlysis (brain hemorrhage)": "Neurolgist",
    "Jaundice": "Hepatologist",
    "Malaria": "Infectious Disease Specialist",
    "Chicken pox": "Pediatrician",
    "Typhoid": "General Physician",
    "Hepatitis A": "Hepatologist",
    "Alcoholic hepatitis": "Gastroenterologist",
    "Tuberculosis": "Pulmonologist",
    "Pneumonia": "Pulmonologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Osteoarthritis": "Rheumatologist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist"
}

# Initialize chatbot LLM
llm = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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" # Set your OpenRouter API key in environment variables
)

system_message = "You act like a healthcare assistant. no need to say it just answer the question directly."

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

def predict_disease(symptoms, location):
    symptom_list = [sym.strip().lower() for sym in symptoms.split(",")]
    user_symptoms = {col: (1 if col in symptom_list else 0) for col in X.columns}
    input_data = pd.DataFrame([user_symptoms], columns=X.columns)

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data).max() * 100

    precautions = precautions_dict.get(prediction.strip().lower().capitalize(), ["No specific precautions found"])
    precautions_text = "\n".join([f"✅ {p}" for p in precautions])  # Format precautions correctly



    # Get doctor specialization
    doctor_specialization = doctor_map.get(prediction, "General Physician")

    # Get list of doctors based on specialization and location
    doctors_list = get_doctors_practo(doctor_specialization, location)

    doctors_list = get_doctors_practo(doctor_specialization, location)
    doctortext = "\n".join([f"✅ {d}" for d in doctors_list])

    return f"Predicted Disease: {prediction}\nChance of Occurrence: {prediction_proba:.2f}%\nPrecautions:\n {precautions_text}%",f"\nDoctors:\n {doctortext}%"

    # return f"Predicted Disease: {prediction}\nChance of Occurrence: {prediction_proba:.2f}%\nPrecautions:\n{precautions_text}", doctors_list    

#Doctor data scrape
def get_doctors_practo(predicted_disease, location="Mumbai"):
    url = f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{predicted_disease}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city={location}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, verify=False)
    if response.status_code != 200:
        return "Failed to fetch data. Try again later."

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all doctor listings
    doctors = soup.find_all("div", class_="info-section")  

    results = []
    for doctor in doctors[:5]:  # Fetch top 5 doctors
        name = doctor.find("h2").text.strip() if doctor.find("h2") else "Unknown"
        specialization_tag = doctor.find("div", class_="u-d-flex")
        specialization = specialization_tag.text.strip() if specialization_tag else "Specialization not available"


        experience = doctor.find("div", class_="uv2-spacer--xs-top").text.strip() if doctor.find("div", class_="uv2-spacer--xs-top") else "experience not available"

        # Extract locality and city using data-qa-id attributes
        locality_tag = doctor.find("span", {"data-qa-id": "practice_locality"})
        city_tag = doctor.find("span", {"data-qa-id": "practice_city"})

        locality = locality_tag.text.strip() if locality_tag else "Locality not available"
        city = city_tag.text.strip() if city_tag else location  # Default to given location

        address = f"{locality} {city}" if locality and city else "Address not available"
        
        results.append(f" 👨‍⚕️ {name}\n {specialization}\t {experience}\n {address}")

    return results if results else "No doctors found."

# Dark Theme using Gradio's Theme System
dark_theme = gr.themes.Base(
    primary_hue="slate",
    secondary_hue="zinc"
).set(
    body_background_fill="#121212",  # Dark background
    body_text_color="black",
    block_background_fill="#1E1E1E",
    block_border_color="#444",
   button_primary_background_fill="#007BFF",  # Blue button
    button_primary_text_color="white",  # White text on button
    # markdown_text_color="white"  # White markdown text

)

custom_css = """
#custom_markdown h1, 
#custom_markdown h2, 
#custom_markdown h3, 
#custom_markdown p {
    color: white !important;
}
#predict_button {
    font-size: 26px !important;
    font-weight: bold;
}
"""

with gr.Blocks(theme = dark_theme, css=custom_css) as interface1:
    gr.Markdown("### 🩺 AI Disease Predictor",elem_id="custom_markdown" )
    with gr.Row():
        symptoms_input = gr.Textbox(label="Enter Symptoms", placeholder="e.g., fever, cough, headache" )
        location_input = gr.Textbox(label="Enter The Location", placeholder="e.g, mumbai, dehli")
        # doctor_text = gr.Textbox(label="Recommended Doctors")
        predict_button = gr.Button("Predict Disease", elem_id="predict_button")
    with gr.Row():
        output_text = gr.Textbox(label="Prediction Result")
        doctor_text = gr.Textbox(label="Recommended Doctors", lines=5)  # Allow multiline display
    predict_button.click(predict_disease, inputs=[symptoms_input, location_input], outputs=[output_text, doctor_text])
   

    

    gr.Markdown("### 🩺 Healthcare Assistant Chatbot",elem_id="custom_markdown" )
    a = gr.ChatInterface(fn=stream_response)
    load_dotenv()


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


interface1.launch(share=True)