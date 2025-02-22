# import requests
# from bs4 import BeautifulSoup

# def get_doctors_practo(predicted_disease, location="Mumbai"):
#     url = f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{predicted_disease}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city={location}"
#     headers = {"User-Agent": "Mozilla/5.0"}

#     response = requests.get(url, headers=headers)
#     if response.status_code != 200:
#         return "Failed to fetch data. Try again later."

#     soup = BeautifulSoup(response.text, "html.parser")
    
#     # Find all doctor listings
#     doctors = soup.find_all("div", class_="info-section")  

#     results = []
#     for doctor in doctors[:5]:  # Fetch top 5 doctors
#         name = doctor.find("h2").text.strip() if doctor.find("h2") else "Unknown"
#         specialization = doctor.find("div", class_="u-grey_3-text").text.strip() if doctor.find("div", class_="u-grey_3-text") else "Specialization not available"
#         address = doctor.find("div", class_="u-d-inline-flex").text.strip() if doctor.find("span", data_qa_id_="practice_locality") else "Address not available"
        
#         results.append(f"👨‍⚕️ {name}\n🔹 {specialization}\n📍 {address}")

#     return results if results else "No doctors found."

# # Example Usage
# predicted_disease = "Dermatologist"
# location = "mumbai"
# doctors_list = get_doctors_practo(predicted_disease, location)
# print("\n\n".join(doctors_list))

import requests
from bs4 import BeautifulSoup

def get_doctors_practo(predicted_disease, location="Mumbai"):
    url = f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{predicted_disease}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city={location}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
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
        
        results.append(f"👨‍⚕️ {name}\n🔹 {specialization}\t {experience}\n📍 {address}")

    return results if results else "No doctors found."

# Example Usage
predicted_disease = "Dermatologist"
location = "Mumbai"
doctors_list = get_doctors_practo(predicted_disease, location)
print("\n\n".join(doctors_list))
