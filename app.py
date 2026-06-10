import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
# CORS ensures your Flutter app (mobile or web) is allowed to talk to this API
CORS(app)

# 📚 Training Dataset
training_sentences = [
    "The air conditioner in classroom 4B is broken and leaking water",
    "Professor did not show up to the lecture hall today",
    "The registration fee payment gateway keeps failing during checkout",
    "There are no lights working in the campus parking lot at night",
    "I need to appeal my final exam marks for computer science",
    "The scholarship application submission portal is closed early",
    "The toilet flusher on the third floor block C is stuck",
    "Timetable scheduling conflicts with my core academic courses",
    "Wi-Fi connection is down in the residential college building"
    "My tuition fee payment went through but my student portal still says unpaid",
    "The bursary office hasn't updated my scholarship status on the system",
    "I need to submit a formal request for a double payment refund"
]

training_labels = [
    "Facility",       # AC leaking
    "Academic",       # Missing lecturer
    "Administrative", # Payment gate
    "Facility",       # Parking lights
    "Academic",       # Grade appeal
    "Administrative", # Scholarship portal
    "Facility",       # Toilet broken
    "Academic",       # Timetables
    "Facility"        # Wi-Fi down
    "Administrative", # Tuition payment status
    "Administrative", # Bursary scholarship status
    "Administrative"  # Double payment refund
]

# 🧠 Train the Naive Bayes Engine immediately when the server starts
print("🤖 Training Naive Bayes Classification Model...")
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(training_sentences, training_labels)
print("✅ Model trained successfully and ready!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter field'}), 400
        
        user_complaint = data['text']
        
        # 🔮 Run Naive Bayes prediction
        predicted_category = model.predict([user_complaint])[0]
        
        return jsonify({
            'status': 'success',
            'predicted_category': predicted_category
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Binds port dynamically for cloud providers like Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
