import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
CORS(app)

# 📚 Expanded High-Density Training Dataset (+20 items per category)
training_sentences = [
    # --- Original Base Dataset ---
    "The air conditioner in classroom 4B is broken and leaking water",
    "Professor did not show up to the lecture hall today",
    "The registration fee payment gateway keeps failing during checkout",
    "There are no lights working in the campus parking lot at night",
    "I need to appeal my final exam marks for computer science",
    "The scholarship application submission portal is closed early",
    "The toilet flusher on the third floor block C is stuck",
    "Timetable scheduling conflicts with my core academic courses",
    "Wi-Fi connection is down in the residential college building",
    "My tuition fee payment went through but my student portal still says unpaid",
    "The bursary office hasn't updated my scholarship status on the system",
    "I need to submit a formal request for a double payment refund",

    # --- Category 1: Facility Expansion (+20) ---
    "The wooden door handle is completely broken and snapped off the frame",
    "Glass window pane in hostel room 204 is cracked and dangerous",
    "Ceiling fan in the discussion room squeaks loudly and rotates too slowly",
    "Water pipe burst in the ground floor bathroom causing minor flooding",
    "The elevator in Block B is stuck on level 3 and doors won't open",
    "No running water inside the female restroom near the main cafeteria",
    "The study table in the library corner has broken wobbly legs",
    "Fluorescent light bulbs are flickering non stop in the laboratory",
    "Wall paint is peeling badly and mold is growing due to moisture dampness",
    "The gym treadmill electronic display screen is broken and won't turn on",
    "Main entrance electronic security gate card scanner is unresponsive",
    "Potholes on the road leading up to the residential college parking space",
    "The sink drain pipe is clogged up with dirt and overflowing with sewage",
    "Air ventilation unit in the seminar hall is blowing hot stale air",
    "Water dispenser on floor 2 is dispensing completely warm water instead of cold",
    "The staircase handrail is loose and detached from the concrete wall",
    "Roof tiling is leaking rainwater directly over the computer lab computers",
    "The main projection screen in auditorium hall A is jammed and won't roll down",
    "Grass is overgrown and obstructing the pedestrian walking paths near blocks",
    "Window latch lock is broken and cannot be closed securely from inside",

    # --- Category 2: Academic Expansion (+20) ---
    "The lecturer is uploaded the wrong assignment sheet template file on portal",
    "I haven't received my supervisor allocation list for my final year project",
    "The course outline syllabus document does not match the actual lecture path",
    "Requesting an extension for the software engineering project submission date",
    "My exam answer booklet grading check contains an arithmetic addition error",
    "Prerequisite subject requirements are not updating despite passing previous course",
    "The lecture slides are completely illegible and missing core reading text",
    "I want to change my major elective course selection option before next week",
    "Professor is refusing to give feedback or review on draft theses papers",
    "Class attendance data tracking sheet shows me absent when I attended",
    "The group project assessment rubric breakdown is highly unclear and vague",
    "Requesting a deferment for my upcoming mid semester examinations due to health",
    "Online quiz portal closed five minutes earlier than specified schedule timeline",
    "The credit hour transfer validation processing for my credit exemption is delayed",
    "Lecturer did not state the mandatory textbook edition list for current course",
    "I cannot access the recorded lecture videos on the learning system library",
    "The deadline for drops and adds period is conflicting with my advisor meeting",
    "My name is missing from the official registered student list for math class",
    "I need an official recommendation letter signed by the academic dean office",
    "The faculty board has not announced the rescheduled final presentation date",

    # --- Category 3: Administrative Expansion (+20) ---
    "Student ID smartcard printing machine is broken at the administrative desk",
    "My internal residential accommodation hostel allocation status is still processing",
    "The finance counter closed earlier than the posted working hours today",
    "I paid my insurance fee but no digital confirmation receipt was emailed",
    "The online registration system keeps crashing with a database error code",
    "I need an official certified true copy statement of my academic transcript",
    "The vehicle parking sticker application form processing time is too long",
    "The automated helpline system bot keeps rejecting my login credentials status",
    "Bursary office deduction amounts for my loan do not match statements",
    "I need to update my personal contact telephone number across campus profiles",
    "The student union election voting link is corrupted and gives a 404 error",
    "Submitting a formal complaint against the rude staff attitude at counter 3",
    "My international student visa renewal documentation confirmation tracking is delayed",
    "The graduation robe collection appointment slots are fully booked out early",
    "Requesting a waiver for late registration penalty fees due to hospital stay",
    "The scholarship funds have not been credited into my bank account yet",
    "I need to cancel my hostel room booking and request deposit refund tracking",
    "The digital document upload section for identity verification keeps failing to load",
    "The financial aid emergency loan processing unit hasn't replied to emails",
    "Lost and found management center doesn't have an item tracking system ledger"
]

training_labels = [
    # --- Base Labels ---
    "facility", "academic", "administrative", "facility", "academic", "administrative",
    "facility", "academic", "facility", "administrative", "administrative", "administrative",

    # --- Facility Labels (+20) ---
    "facility", "facility", "facility", "facility", "facility", "facility", "facility",
    "facility", "facility", "facility", "facility", "facility", "facility", "facility",
    "facility", "facility", "facility", "facility", "facility", "facility",

    # --- Academic Labels (+20) ---
    "academic", "academic", "academic", "academic", "academic", "academic", "academic",
    "academic", "academic", "academic", "academic", "academic", "academic", "academic",
    "academic", "academic", "academic", "academic", "academic", "academic",

    # --- Administrative Labels (+20) ---
    "administrative", "administrative", "administrative", "administrative", "administrative",
    "administrative", "administrative", "administrative", "administrative", "administrative",
    "administrative", "administrative", "administrative", "administrative", "administrative",
    "administrative", "administrative", "administrative", "administrative", "administrative"
]

# 🧠 Train the Naive Bayes Engine immediately when the server starts
print("🤖 Training Expanded Naive Bayes Classification Model...")
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(training_sentences, training_labels)
print(f"✅ Model trained successfully with {len(training_sentences)} active data points!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter field'}), 400
        
        user_complaint = data['text']
        predicted_class = model.predict([user_complaint])[0]
        
        return jsonify({
            'status': 'success',
            'category': str(predicted_class),  
            'prediction': str(predicted_class)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
