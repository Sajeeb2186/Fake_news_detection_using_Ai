import tkinter as tk
from tkinter import messagebox
import pickle
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

def clean_input(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = stopwords.words('english')
    return ' '.join([word for word in text.split() if word not in stop_words])

# ✅ Load SVM Model and Vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vector.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict():
    user_input = entry.get("1.0", tk.END).strip()
    
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    
    cleaned_text = clean_input(user_input)
    vect_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(vect_input)[0]

    result_text = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    result_color = "#e74c3c" if prediction == 1 else "#27ae60"
    label_result.config(text=result_text, fg=result_color)

app = tk.Tk()
app.title("Fake News Detector (SVM)")
app.geometry("720x550")
app.configure(bg='#f0f3f4')

title_font = ("Arial", 24, "bold")
input_font = ("Helvetica", 10)
button_font = ("Arial", 12, "bold")
result_font = ("Arial", 14, "bold")

# Header Section
header_frame = tk.Frame(app, bg='#2c3e50')
header_frame.pack(fill='x')

tk.Label(
    header_frame,
    text="Fake News Detection System (SVM)",
    font=title_font,
    bg='#2c3e50',
    fg='white',
    pady=20
).pack()

# Input Section
input_frame = tk.Frame(app, bg='#f0f3f4')
input_frame.pack(pady=20)

tk.Label(
    input_frame,
    text="Enter News Article:",
    font=("Helvetica", 12, "bold"),
    bg='#f0f3f4',
    fg='#2c3e50'
).pack(anchor='w', pady=5)

entry = tk.Text(
    input_frame,
    height=12,
    width=75,
    font=input_font,
    bd=2,
    relief='solid',
    padx=10,
    pady=10
)
entry.pack()

# Button Section
button_frame = tk.Frame(app, bg='#f0f3f4')
button_frame.pack(pady=15)

analyze_btn = tk.Button(
    button_frame,
    text="Analyze Text",
    command=predict,
    font=button_font,
    bg='#3498db',
    fg='white',
    activebackground='#2980b9',
    relief='flat',
    padx=30,
    pady=8
)
analyze_btn.pack()

result_frame = tk.Frame(app, bg='#f0f3f4')
result_frame.pack(pady=15)

label_result = tk.Label(
    result_frame,
    text="",
    font=result_font,
    bg='#f0f3f4',
    pady=10,
    padx=20
)
label_result.pack()

app.mainloop()
