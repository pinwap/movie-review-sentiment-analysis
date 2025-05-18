import os #file
import re #regular expression ใช้ค้นหาคำ
import string #string
import time #time
import itertools
import logging #เก็บขั้นตอนการทำงาน
import numpy as np #array

logging.basicConfig(
    filename='pin.log', #ชื่อไฟล์ที่เก็บ log
    level=logging.DEBUG, #ระดับการเก็บ log
    format='%(asctime)s - %(levelname)s - %(message)s', #รูปแบบการเก็บ log
    filemode='w' #overwrite log file
)

def print_log(level, message):
  if level.lower() == "info":
    logging.info(message)
  elif level.lower() == "warning":
    logging.warning(message)
  elif level.lower() == "error":
    logging.error(message)
  elif level.lower() == "debug":
    logging.debug(message)
  else:
    logging.info(message)
  print(message)


logging.info("Script started for text classification.") #เก็บ log ว่าเริ่มทำงาน
print("Script started for text classification. Logging to training_log.log") #แสดงข้อความว่าเริ่มทำงานขึ้น terminal

#---------------------
# GPU configuration การ์ดจอ
#-----------------------

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU') #list GPU ที่มีอยู่
if gpus:
    logging.info(f"GPU available: {gpus}") #ถ้ามี GPU ให้เก็บ log ว่ามี GPU
    print(f"GPUs Available: {gpus}") #แสดงข้อความว่า GPU available
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) #set memory growth
            logging.info(f"Set memory growth for GPU: {gpu}") #เก็บ log ว่า set memory growth
            print(f"Set memory growth for GPU: {gpu}") #แสดงข้อความว่า set memory growth
    except RuntimeError as e:
        logging.error(f"Error setting memory growth: {e}") #ถ้าเกิด error ให้เก็บ log ว่า error
        print(f"Error setting memory growth: {e}") #แสดงข้อความว่า error
else:
    logging.info("No GPUs available.")
    print("No GPUs available.")

#-----------------------
# Output directory for models
#-----------------------
output_folder = 'output' #output folder
os.makedirs(output_folder, exist_ok=True) #สร้าง folder ถ้ายังไม่มี
logging.info(f"Output directory created: {output_folder}") #เก็บ log ว่าสร้าง folder เรียบร้อย
print(f"Output directory created: {output_folder}") #แสดงข้อความว่าสร้าง folder เรียบร้อย

#-----------------------
# Download dataset
#-----------------------
# "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
try:
    import kagglehub #library รวมดาต้าเซต
    print("Downloading dataset from KaggleHub...")
    train_dataset_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews") #ดาวน์โหลดดาต้าเซต
    logging.info(f"Dataset downloaded from KaggleHub: {train_dataset_path}") #เก็บ log ว่าดาวน์โหลดดาต้าเซตเรียบร้อย
    print(f"Dataset downloaded from KaggleHub: {train_dataset_path}") #แสดงข้อความว่าดาวน์โหลดดาต้าเซตเรียบร้อย
except ImportError:
    logging.error("KaggleHub not installed. Please install it to download datasets.")
    print("KaggleHub not installed. Please install it to download datasets.") #แสดงข้อความว่าไม่สามารถดาวน์โหลดดาต้าเซตได้
    train_dataset_path = './lakshmi25npathi/imdb-dataset-of-50k-movie-reviews'
    if not os.path.exists(train_dataset_path):
        logging.error(f"Dataset path {train_dataset_path} does not exist. Exiting.")
        print(f"Dataset path {train_dataset_path} does not exist. Exiting")
        exit()

except Exception as e:
    logging.error(f"Error downloading training dataset: {e}")
    print(f"Error downloading training dataset: {e}")
    exit()


train_file = os.path.join(train_dataset_path, 'IMDB Dataset.csv') #ดูว่าไฟล์ train.csv มีมั้ย อยู่ที่ไหน
if not os.path.exists(train_file): #ถ้าไม่มีไฟล์ train.csv
    print_log("error", f"CSV file not found at {train_file}") #แสดงข้อความว่าไม่พบไฟล์ train.csv
    exit() #exit program

import pandas as pd
data = pd.read_csv(train_file)
print_log("info", f"Training dataset loaded with shape: {data.shape}") #เก็บ log ว่าข้อมูลที่โหลดมาเป็นยังไง แกนx,yมีขนาดเท่าไหร่ = มีกี่คอลัมน์กี่แถว

#show 5 sample data
print(data.head(5)) #แสดงข้อมูล 5 แถวแรก

# -----------------------
# Data Preprocessing
# -----------------------

# ลบ stop word : คำที่ไม่สื่อความหมายใน NLP เช่นคำว่า "the", "is", "in" เป็นต้น
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') #ดาวน์โหลด stop words

def preprocess_text(text):
    text = text.lower() #แปลงเป็นตัวพิมพ์เล็กทั้งหมด
    text = re.sub(r'<.*?>', '', text) #ลบ HTML tags
    text = re.sub(r'http\S+|www\S', '', text) #ลบ URL
    text = text.translate(str.maketrans('', '', string.punctuation)) #ลบ punctuation เช่น .,!? เป็นต้น
    text = re.sub(r'\d+', '', text) #ลบตัวเลข
    text = re.sub(r'\s+', ' ', text).strip() #ลบ whitespace
    return text

data['clean_review'] = data['review'].apply(preprocess_text) #ใช้ฟังก์ชัน preprocess_text กับคอลัมน์ review
stop_words = set(stopwords.words('english')) #โหลด stop words
data['clean_review'] = data['clean_review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words])) #ลบ stop words
data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0}) #แปลง label เป็น 1 และ 0
print_log("info", f"Data preprocessing completed. Sample data: {data.head(5)}") #เก็บ log ว่าทำการ preprocess ข้อมูลเสร็จเรียบร้อยแล้ว

X = data['clean_review'] #X คือ คอลัมน์สำหรับเทรนเอไอ
y = data['label'] #y คือ คอลัมน์เฉลยที่ต้องการให้ AI ทำนาย


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) #แบ่งข้อมูลเป็น 80% สำหรับเทรน 20% สำหรับ validation
print_log("info", f"Training and validation data split. Training size: {len(X_train)}, Validation size: {len(X_val)}")

#-----------------------
# Helper functions
#-----------------------
import matplotlib.pyplot as plt #plot graph
def save_classification_report(report, model_name, dataset_tag = ""):
    report_path = os.path.join(output_folder, f"{model_name}_classification_report.txt") #สร้าง path สำหรับเก็บ classification report
    with open(report_path, 'w') as f:
        f.write(report) #เขียน classification report ลงไฟล์
    print_log("info",f"{model_name}{dataset_tag} classification report saved to {report_path}") #เก็บ log ว่า save classification report เรียบร้อยแล้ว

def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(8, 6)) #สร้าง figure ขนาด 8x6
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) #แสดง confusion matrix
    plt.title(title) #ตั้งชื่อ confusion matrix
    plt.colorbar() #แสดง color bar
    tick_marks = np.arange(len(classes)) #สร้าง tick marks
    plt.xticks(tick_marks, classes, rotation=45) #ตั้งชื่อ x-axis
    plt.yticks(tick_marks, classes) #ตั้งชื่อ y-axis
    thresh = cm.max() / 2. #หาค่า threshold
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #loop ผ่าน confusion matrix
        plt.text(j, i, format(cm[i, j], 'd'), #แสดงค่าใน confusion matrix
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label') #ตั้งชื่อ y-axis
    plt.xlabel('Predicted label') #ตั้งชื่อ x-axis
    plt.tight_layout() #จัดระเบียบ layout
    plt.savefig(save_path) #save confusion matrix
    logging.info(f"Confusion matrix saved to {save_path}")# เก็บ log ว่า save confusion matrix เรียบร้อยแล้ว
    plt.close() #close figure

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    history_plot_path = os.path.join(output_folder, f"{model_name}_training_history.png")
    plt.savefig(history_plot_path)
    logging.info(f"{model_name} training history plot saved to {history_plot_path}")
    plt.close()

#-----------------------
# Model training
#-----------------------
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model_name = "LSTM_text_classification" #long short term memory
max_word = 10000 #จำนวนคำสูงสุดที่ใช้ในการเทรน
max_len = 200 #จำนวนคำสูงสุดในแต่ละประโยค

tokenizer = Tokenizer(num_words=max_word) #ตัวตัดคำ
tokenizer.fit_on_texts(X_train) #fit tokenizer กับข้อมูลเทรน

X_train_seq = tokenizer.texts_to_sequences(X_train) #แปลงข้อมูลเทรนเป็น sequence
X_val_seq = tokenizer.texts_to_sequences(X_val) #แปลงข้อมูล validation เป็น sequence

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len) #padding ข้อมูลเทรน
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len) #padding ข้อมูล validation

lstm_model = Sequential(
    [
        Embedding(input_dim=max_word, output_dim=128, input_length=max_len),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid') #1 = output dimension
    ], name=model_name
)

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.summary() #แสดงสรุปโมเดล

print_log("info", lstm_model.summary()) #เก็บ log ว่าสรุปโมเดลเป็นยังไง

#-----------------------
# Model training
#-----------------------

print_log("info", "Training LSTM model...") #เก็บ log ว่าเริ่มเทรนโมเดล
start_time = time.time() #เริ่มนับเวลา

lstm_history = lstm_model.fit(X_train_pad, y_train, epochs=6, batch_size=8192, validation_data=(X_val_pad, y_val), verbose=1)

train_time = time.time() - start_time #คำนวณเวลาในการเทรน
print_log("info", f"LSTM model training completed in {train_time:.2f} seconds.") #เก็บ log ว่าเทรนโมเดลเสร็จเรียบร้อยแล้ว

# -----------------------
# **NEW**: Save Model & Tokenizer
# -----------------------
model_path = os.path.join(output_folder, f"{model_name}.h5")
tokenizer_path = os.path.join(output_folder, "tokenizer.pkl")
lstm_model.save(model_path)

import pickle
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Saved model to {model_path}")
print(f"Saved tokenizer to {tokenizer_path}")

#-----------------------
# Model evaluation
#-----------------------

lstm_score_val = lstm_model.evaluate(X_val_pad, y_val, verbose=0) #ได้เป็นlist
lstm_accurency_val = lstm_score_val[1]
# ใส่ข้อมูลที่ใช้วัดผล แล้วดูว่าได้คำตอบตรงแค่ไหน

y_pred_lstm_val = (lstm_model.predict(X_val_pad) > 0.5).astype("int32")
report_lstm_val = classification_report(y_val, y_pred_lstm_val, digits=3) #classification report
print_log("info", f"LSTM model validation accuracy: {report_lstm_val}") #เก็บ log ว่า accuracy เท่ากับเท่าไหร่
print_log("info", report_lstm_val) #เก็บ log ว่า classification report เป็นยังไง
save_classification_report(report_lstm_val, model_name, dataset_tag = "_val") #save classification report
plot_training_history(lstm_history, model_name)

cm_lstm_val = confusion_matrix(y_val, y_pred_lstm_val)
cm_lstm_val_path = os.path.join(output_folder, f"{model_name}_confusion_matrix_val.png")
plot_confusion_matrix(cm_lstm_val, classes=['Negative', 'Positive'], title='LSTM Model Validation Confusion Matrix', save_path=cm_lstm_val_path)

