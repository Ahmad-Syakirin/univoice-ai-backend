import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
CORS(app)

# =========================================================================
# --- CATEGORY 1: FACILITY SAMPLES ----------------------------------------
# =========================================================================
facility_sentences = [
    "The air conditioner in classroom 4B is broken and leaking water",
    "There are no lights working in the campus parking lot at night",
    "The toilet flusher on the third floor block C is stuck",
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
    "The pathway concrete tiles have cracked due to growing tree root pressure underneath"
]

# =========================================================================
# --- CATEGORY 2: ACADEMIC SAMPLES ----------------------------------------
# =========================================================================
academic_sentences = [
    "Professor did not show up to the lecture hall today",
    "I need to appeal my final exam marks for computer science",
    "Timetable scheduling conflicts with my core academic courses",
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
    "The deadline for drops and adds period is conflicting with my academic advisor meeting",
    "My name is missing from the official registered student list for math class",
    "I need an official recommendation letter signed by the academic dean office",
    "The faculty board has not announced the rescheduled final presentation date",
    "I need to submit an official exam paper grading appeal because my total marks were miscalculated",
    "The final exam schedule lists two core academic exams at the exact same hour on Monday morning",
    "The mid semester exam question sheet contained multiple formatting errors that made questions unreadable",
    "Requesting a formal re evaluation of my computer architecture final exam paper marks",
    "The professor refuses to give any breakdown or feedback regarding our recent midterm exam grades",
    "The final examination timetable clashes directly with my other core course presentation schedule",
    "I was falsely marked absent during the accounting final exam even though I signed the attendance sheet",
    "The online examination portal locked me out ten minutes before the scheduled submission deadline expired",
    "I need an extension on my software engineering assignment because the professor altered the rubric late",
    "The course outline syllabus does not match the actual topics covered in the weekly lectures",
    "We have not been assigned our final year project supervisor yet despite classes starting weeks ago",
    "The group project assessment rubric structure is completely vague and lacks clear marking parameters",
    "Requesting a course deferment for the upcoming final exams due to documented medical emergency circumstances",
    "The credit hour transfer validation processing for my prerequisite math exemptions is still delayed",
    "The lecturer failed to provide the mandatory reference textbook reading list for this semester course",
    "I am unable to view or access the recorded lecture videos on the university digital learning management system",
    "The add drop course registration deadline conflicts directly with my mandatory academic advisor meeting slot",
    "My student profile is entirely missing from the official registered student roster for the calculus class",
    "I need an official recommendation letter signed by the academic dean for my internship application",
    "The faculty academic board has still not released the rescheduled date for our deferred project presentations",
    "The assignment submission slot on the student portal is completely locked down and won't accept files",
    "Our group requires a brief deadline extension to complete the data collection for the research paper assignment",
    "The lecturer refuses to explain why my lab report lost points during the final marking review",
    "I want to change my current major elective course selection option before the engineering module closes",
    "I cannot find the grading rubric framework anywhere for the essay task that is due next week",
    "The virtual lecture room link for our Friday morning academic class is broken and inaccessible",
    "The faculty altered our core module textbook requirements without informing any of the enrolled students",
    "I attended every mandatory tutorial session but my grade book tracking displays zero percent progress",
    "The prerequisite rule blocks me from adding advanced data structures to my current semester academic schedule",
    "I would like to drop my accounting minor module and switch over to the statistics pathway option",
    "The faculty office has failed to post our mid term quiz results on the public tracking board",
    "The academic instructions provided for our physics laboratory assignment are completely contradictory",
    "I need an academic transcript review because an exempted subject is still marked as unfulfilled",
    "Our lecturer reads directly from the presentation slides and refuses to answer academic queries during class",
    "The dean has not approved my official request to take extra credit hours this term",
    "I missed the unannounced pop quiz because the notification went live only ten minutes before class started",
    "The final course presentation list completely excludes our project group number from the roster schedule",
    "The textbook requirements list a course edition that is completely out of print and unavailable to buy",
    "The digital learning library database is missing the supplementary reading chapters for the final exam block",
    "My academic grades achieved during my exchange semester abroad have not been integrated into my profile",

    # 🪄 ADDED: Academic Portal context tokens to fix the cross routing bug
    "The assignment rubric is missing from the student portal layout",
    "My lecturer has not uploaded the assignment rubric on the student portal yet",
    "Lecturer posted the wrong quiz configuration files on the course portal",
    "The student portal is not showing my course evaluation syllabus or handouts",
    "I cannot find the grading rubric structure anywhere on the student portal profile"
]

# =========================================================================
# --- CATEGORY 3: ADMINISTRATIVE SAMPLES ----------------------------------
# =========================================================================
administrative_sentences = [
    "The registration fee payment gateway keeps failing during checkout",
    "The scholarship application submission portal is closed early",
    "My tuition fee payment went through but my student portal still says unpaid",
    "The bursary office hasn't updated my scholarship status on the system",
    "I need to submit a formal request for a double payment refund",
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
    "Lost and found management center doesn't have an item tracking system ledger",
    "My student account login token has expired and I cannot reset my profile password",
    "The automated teller system at the student bank counter ate my smart registration card",
    "I need to apply for a campus parking decal sticker for my new motorcycle",
    "The bursary department added a late payment fine to my file even though I paid on time",
    "I have been waiting over three weeks for the admin office to process my change of details",
    "The online queue management system for counter services keeps dropping my ticket number",
    "I need to request an official letter proving my enrollment status for insurance purposes",
    "The international student help desk lost my passport copies for the clearance check",
    "The refund processing portal tells me my banking information format is invalid when it isn't",
    "I cannot pay my graduation fees because the online bill option is missing from my page",
    "The front counter staff was incredibly dismissive when I went to ask about hostel keys",
    "My student visa extension application has been stuck on level one verification for a month",
    "The orientation package collection room was closed during its scheduled operating hours",
    "I need a certified true copy of my foundation certificate from the central archives",
    "The system won't let me sign up for campus housing even though I am a first year student",
    "I need to update my emergency contact details but the data edit button is greyed out",
    "The finance office is demanding a physical bank receipt when I already sent the transaction PDF",
    "The student union ballot system allowed people to vote multiple times due to a software glitch",
    "I was charged an extra amenity fee on my bill but I don't live in the main campus dorms",
    "The digital verification link emailed to me keeps saying page expired whenever I open it",
    "I need to file a formal dispute against a wrongful financial hold placed on my account",
    "The helpdesk support team hasn't responded to my login trouble ticket in over five days",
    "The administrative directory listing has the wrong office numbers for the welfare team",
    "I cannot print out my library clearance certificate because of an unverified balance of zero",
    "The scholarship department sent my payout check to the wrong residential mailing address",
    "I need to cancel my meal plan subscription and get the balance credited back to my wallet",
    "The portal keeps looping back to the login screen whenever I click on the payment button",
    "The main registry desk refuses to accept my medical certificate document for fee waiver",
    "I am trying to submit my health insurance declaration form but the file upload limit is too low",
    "The locker rental signup system took my money but didn't assign me a locker unit number",
    "My student profile shows the completely wrong national identity card number on the main page",
    "The counter queue app does not send a notification when your turn is coming up next",
    "I need to collect my original school certificates from the admissions filing vault room",
    "The bursary office hasn't updated my sponsored account status so I can't register for classes",
    "I am trying to officially withdraw from this semester but the digital drop form won't submit",
    "The administration building information board has outdated application deadlines listed",
    "I need an official statement from the finance team breaking down my outstanding fees balance",
    "The campus shuttle bus pass registration system keeps rejecting my uploaded portrait photo",
    "The student support counter has only one staff working during peak lunch hour rush timing",
    "I paid my club registration fees but my name isn't appearing on the official society ledger",
    "The portal gave me a success message for housing but I haven't received my room number assignment",
    "I need to clear an administrative block on my account that was placed there by mistake"
]

# 🔗 Combine all datasets seamlessly
training_sentences = facility_sentences + academic_sentences + administrative_sentences

# 🏷️ Bulletproof Automatic Matching Logic
training_labels = (
    ["facility"] * len(facility_sentences) +
    ["academic"] * len(academic_sentences) +
    ["administrative"] * len(administrative_sentences)
)

# 🧠 Train the Naive Bayes Engine immediately when the server starts
print(f"🤖 Training Model with {len(training_sentences)} sentences and {len(training_labels)} labels...")
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(training_sentences, training_labels)
print("✅ Symmetrical Model trained successfully!")

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
            'predicted_category': str(predicted_class)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
