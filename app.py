import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
CORS(app)

# 📚 Expanded Training Dataset (with 40 new explicit Facility anchors)
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

    # --- Category 1: Facility Base Expansion ---
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

    # --- 🚨 CRITICAL MASS FACILITY INJECTION (+40 NEW TARGET SAMPLES) ---
    "The door lock is jammed and I am locked out of my hostel room",
    "The lock on the bathroom door is broken and cannot be secured",
    "Front entrance main wooden door is sagging off its hinges and won't close",
    "The classroom door knob keeps spinning completely loose and won't unlatch",
    "Glass panels on the corridor emergency exit doors are completely shattered",
    "The back door to the lecture theatre is squeaking incredibly loudly during use",
    "Padlock on the sports equipment storage locker room is rusted shut",
    "The main gate iron hinges are broken making it dangerous to pull open",
    "Water is dripping constantly from the ceiling panel in the main hallway",
    "Bathroom tap pipe is leaking a pool of water underneath the common basin",
    "The water pressure in the residential shower blocks is way too low to use",
    "Toilet bowl is completely blocked up and overflowing onto the tiles",
    "The main drainage system outside the dining hall is clogged with leaves",
    "Borehole water pump is making a loud grinding grinding sound and failing",
    "Water heater in the hostel bathroom is completely dead and only shoots cold water",
    "A major pipe link under the library foundation is cracked and shooting water",
    "The main fuse box breaker keeps tripping every time we turn on lab equipment",
    "Exposed electrical wires are hanging down from the broken hallway light fixture",
    "Wall power sockets and plugs in study cubicle row B have no electricity",
    "The presentation room backup generator is smoking and leaking oil engine fluid",
    "Streetlights along the pathway between the block and library are completely dark",
    "The exhaust ventilation extraction fan in the chemistry lab has burnt out",
    "Electrical sparks fly whenever we attempt to plug something into the wall socket",
    "The main internet server room air conditioning unit has failed and overheated",
    "The ceiling plasterboard panels are soft sagging down and about to collapse",
    "Cracks are beginning to open up along the concrete support beams of the balcony",
    "Floor tiles are loose broken and lifting up in the high traffic walkway corridor",
    "The window frame has warped from sun damage and cannot be shut tightly",
    "The brick wall surrounding the campus perimeter has partly collapsed near the gate",
    "The carpet in the auditorium is completely torn up and creating a tripping hazard",
    "The library study desk chair backrest has completely snapped off from the metal base",
    "The whiteboard mounting bracket has slipped out of the wall and is hanging crooked",
    "Blinds on the classroom windows are jammed pulled sideways and cannot be adjusted",
    "The projector mount is vibrating violently when the ceiling fan is switched on",
    "The microwave appliance inside the student lounge area sparks and smokes when used",
    "The cafeteria public refrigerator unit is no longer cold and food is spoiling",
    "The washing machine unit number 3 inside the laundry room is leaking soapy water",
    "The computer lab emergency fire alarm bell is hanging off its wire harness",
    "Fire extinguisher unit bracket has broken and the canister is rolling on the floor",
    "The pathway concrete tiles have cracked due to growing tree root pressure underneath",

    # --- Category 2: Academic Expansion ---
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

    # --- Category 3: Administrative Expansion ---
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

    # --- Facility Base Labels ---
    "facility", "facility", "facility", "facility", "facility", "facility", "facility",
    "facility", "facility", "facility", "facility", "facility", "facility", "facility",
    "facility", "facility", "facility", "facility", "facility", "facility",

    # --- 🚨 INJECTED FACILITY TARGET LABELS (Exactly 40 Elements) ---
    "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility",
    "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility",
    "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility",
    "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility", "facility",

    # --- Academic Labels ---
    "academic", "academic", "academic", "academic", "academic", "academic", "academic",
    "academic", "academic", "academic", "academic", "academic", "academic", "academic",
    "academic", "academic", "academic", "academic", "academic", "academic",

    # --- Administrative Labels ---
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
            'predicted_category': str(predicted_class) # Kept both for safety across variations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
