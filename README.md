# Telekom Churn Analysis

## Loyiha maqsadi
Telekom kompaniyasi mijozlarining ketish ehtimolini machine learning
yordamida bashorat qilish va biznesga amaliy yechimlar taklif qilish.

## Loyiha tuzilishi

telekom_churn_analysis/
├── data/                    # Ma'lumotlar (CSV fayllar)
├── process/                 # Jupyter notebook'lar
│   ├── data_set_cleaning.ipynb
│   └── churn_analysis.ipynb
├── models/                  # Saqlangan ML modellar
├── webapp/                  # Streamlit veb-sayt
│   └── app.py
├── telegram_bot/            # Telegram bot
│   └── bot.py
├── requirements.txt         # Python kutubxonalar
├── .gitignore              # Git ignore qoidalari
└── README.txt              # Bu fayl

## Tezkor boshlash

### 1. Virtual environment yaratish (tavsiya)
python -m venv venv

# Windows uchun:
venv\\Scripts\\activate

# Mac/Linux uchun:
source venv/bin/activate

### 2. Kutubxonlarni o'rnatish
pip install -r requirements.txt

### 3. Ma'lumotlarni tayyorlash
- cleaned_data.csv faylini data/ papkasiga joylang
- jupyter notebook notebooks/data_set_cleaning.ipynb

### 4. Tahlil va modellashtirish
jupyter notebook notebooks/churn_analysis.ipynb

### 5. Veb-saytni ishga tushirish
cd webapp
streamlit run app.py
# Browser'da: http://localhost:8501

### 6. Telegram bot sozlash
1. @BotFather dan token oling
2. telegram_bot/bot.py da YOUR_BOT_TOKEN_HERE ni almashtiring
3. python telegram_bot/bot.py

## Texnologiyalar

- Python 3.8+ - Asosiy dasturlash tili
- Pandas, NumPy - Ma'lumotlar tahlili
- Matplotlib, Seaborn - Vizualizatsiya
- Scikit-learn - Machine Learning
- Streamlit - Veb interfeys
- python-telegram-bot - Telegram bot
- Jupyter - Interactive development

## ML Modellari

1. **Logistic Regression**
   - Tez va sodda
   - Binary classification
   - Kutilayotgan aniqlik: ~80-85%

2. **Random Forest**
   - Ensemble method
   - Feature importance
   - Kutilayotgan aniqlik: ~85-90%

## Loyiha bosqichlari

### 1-bosqich: Ma'lumotlar tozalash
- Missing values ni to'ldirish
- Dublikatlarni olib tashlash
- Ma'lumot turlarini to'g'rilash
- Outliers ni tekshirish

### 2-bosqich: Tahlil va vizualizatsiya
- Exploratory Data Analysis (EDA)
- Churn patterns ni aniqlash
- Statistik testlar

### 3-bosqich: Model yaratish
- Feature engineering
- Train/Test split
- Model training
- Performance evaluation

### 4-bosqich: Deployment
- Streamlit veb-sayt
- Telegram bot
- Model saqlash (PKL)

## Kutilayotgan natijalar

### Asosiy topilmalar:
- Yangi mijozlar (0-12 oy) eng yuqori risk
- Yuqori oylik to'lov ketish ehtimolini oshiradi
- Month-to-month shartnomalar xavfli
- Qo'shimcha xizmatlar mijozlarni saqlab qoladi

### Biznes tavsiyalar:
- Yangi mijozlarga maxsus onboarding dasturi
- Narx strategiyasini qayta ko'rish
- Uzoq muddatli shartnomalarni rag'batlantirish
- Qo'shimcha xizmatlar marketing

## Muammolarni hal qilish

### Keng uchraydigan muammolar:

**1. Jupyter notebook ishlamaydi**
pip install jupyter
jupyter --version

**2. Streamlit ishlamaydi**
pip install streamlit
streamlit hello

**3. CSV fayl yuklanmaydi**
# Fayl yo'lini tekshiring:
import os
print(os.getcwd())
print(os.listdir('data/'))

**4. Model yuklanmaydi**
# Avval churn_analysis.ipynb ni to'liq ishga tushiring

**5. Telegram bot ishlamaydi**
# Bot token to'g'ri kiritilganini tekshiring
# @BotFather dan yangi token oling

## Minnatdorchilik

- Telekom dataset uchun Kaggle community
- Open source kutubxonalar yaratuvchilari
- Machine Learning community
