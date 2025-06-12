import streamlit as st
import pandas as pd
import joblib
import os

# Sahifa sozlamalari
st.set_page_config(
    page_title="Telekom Churn Bashorati",
    page_icon="📱",
    layout="wide"
)

# CSS stillar
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .result-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .result-danger {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Modellarni yuklash"""
    try:
        # Model fayllari yo'li
        model_path = '../models'
        
        # Random Forest modelini yuklash (eng yaxshi deb faraz qilamiz)
        model = joblib.load(f'{model_path}/random_forest.pkl')
        
        # Label encoders yuklash
        label_encoders = joblib.load(f'{model_path}/label_encoders.pkl')
        
        # Feature names yuklash
        feature_names = joblib.load(f'{model_path}/feature_names.pkl')
        
        return model, label_encoders, feature_names
        
    except Exception as e:
        st.error(f"Modellarni yuklashda xatolik: {e}")
        return None, None, None

def predict_churn(model, label_encoders, feature_names, user_input):
    """Churn bashorati"""
    try:
        # Input ma'lumotlarni DataFrame ga aylantirish
        input_df = pd.DataFrame([user_input])
        
        # Kategorik ustunlarni kodlash
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])
        
        # Feature names tartibida joylash
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        # Bashorat qilish
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Bashorat qilishda xatolik: {e}")
        return 0, [0.5, 0.5]

def main():
    # Sarlavha
    st.markdown("""
    <div class="main-header">
        <h1>📱 Telekom Churn Bashorati</h1>
        <p>Mijoz ketish ehtimolini aniqlash tizimi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modellarni yuklash
    model, label_encoders, feature_names = load_models()
    
    if model is None:
        st.error("❌ Modellar yuklanmadi! Iltimos, avval notebook'ni ishga tushirib modellarni yarating.")
        return
    
    st.success("✅ Modellar muvaffaqiyatli yuklandi!")
    
    # Asosiy qism
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👤 Mijoz Ma'lumotlari")
        
        with st.form("prediction_form"):
            # Asosiy ma'lumotlar
            gender = st.selectbox("Jinsi:", ["Male", "Female"])
            
            senior_citizen = st.selectbox("Keksa fuqaro:", [0, 1], 
                                        format_func=lambda x: "Yo'q" if x == 0 else "Ha")
            
            partner = st.selectbox("Sherik:", ["No", "Yes"])
            
            dependents = st.selectbox("Qaramog'idagilar:", ["No", "Yes"])
            
            tenure = st.slider("Xizmat muddati (oylar):", 0, 72, 12)
            
            # Xizmatlar
            st.markdown("**Xizmatlar:**")
            
            phone_service = st.selectbox("Telefon xizmati:", ["No", "Yes"])
            
            multiple_lines = st.selectbox("Ko'p liniyalar:", 
                                        ["No", "Yes", "No phone service"])
            
            internet_service = st.selectbox("Internet xizmati:", 
                                          ["DSL", "Fiber optic", "No"])
            
            online_security = st.selectbox("Onlayn xavfsizlik:", 
                                         ["No", "Yes", "No internet service"])
            
            online_backup = st.selectbox("Onlayn zaxira:", 
                                       ["No", "Yes", "No internet service"])
            
            device_protection = st.selectbox("Qurilma himoyasi:", 
                                           ["No", "Yes", "No internet service"])
            
            tech_support = st.selectbox("Texnik yordam:", 
                                      ["No", "Yes", "No internet service"])
            
            streaming_tv = st.selectbox("TV oqimi:", 
                                      ["No", "Yes", "No internet service"])
            
            streaming_movies = st.selectbox("Film oqimi:", 
                                          ["No", "Yes", "No internet service"])
            
            # Shartnoma va to'lov
            st.markdown("**Shartnoma va To'lov:**")
            
            contract = st.selectbox("Shartnoma turi:", 
                                  ["Month-to-month", "One year", "Two year"])
            
            paperless_billing = st.selectbox("Qog'ozsiz hisob:", ["No", "Yes"])
            
            payment_method = st.selectbox("To'lov usuli:", 
                                        ["Electronic check", "Mailed check", 
                                         "Bank transfer (automatic)", 
                                         "Credit card (automatic)"])
            
            monthly_charges = st.number_input("Oylik to'lov ($):", 
                                            min_value=0.0, max_value=200.0, 
                                            value=65.0, step=0.1)
            
            total_charges = st.number_input("Umumiy to'lov ($):", 
                                          min_value=0.0, max_value=10000.0, 
                                          value=monthly_charges * tenure, step=0.1)
            
            # Bashorat tugmasi
            predict_btn = st.form_submit_button("🔮 Bashorat Qilish", 
                                               use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Bashorat Natijalari")
        
        if predict_btn:
            # Foydalanuvchi ma'lumotlari
            user_input = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Bashorat qilish
            prediction, probability = predict_churn(model, label_encoders, 
                                                  feature_names, user_input)
            
            # Natijalarni ko'rsatish
            churn_prob = probability[1] * 100
            stay_prob = probability[0] * 100
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-danger">
                    <h3>⚠️ Mijoz Ketishi Mumkin</h3>
                    <h2>{churn_prob:.1f}%</h2>
                    <p>Ketish ehtimoli yuqori</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tavsiyalar
                st.markdown("### 💡 Tavsiyalar:")
                st.write("• Mijoz bilan shaxsiy aloqa o'rnating")
                st.write("• Maxsus chegirmalar taklif qiling")
                st.write("• Qo'shimcha xizmatlar ko'rsating")
                
            else:
                st.markdown(f"""
                <div class="result-success">
                    <h3>✅ Mijoz Qoladi</h3>
                    <h2>{stay_prob:.1f}%</h2>
                    <p>Qolish ehtimoli yuqori</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tavsiyalar
                st.markdown("### 💡 Tavsiyalar:")
                st.write("• Mijozni rag'batlantiring")
                st.write("• Yangi xizmatlar taklif qiling")
                st.write("• Uzoq muddatli shartnoma taklif qiling")
            
            # Ehtimollik ko'rsatkich
            st.markdown("### 📊 Ehtimollik Ko'rsatgichi")
            
            # Progress bar
            st.write(f"**Qolish ehtimoli:** {stay_prob:.1f}%")
            st.progress(stay_prob / 100)
            
            st.write(f"**Ketish ehtimoli:** {churn_prob:.1f}%")
            st.progress(churn_prob / 100)
            
        else:
            st.info("👈 Chap tarafda mijoz ma'lumotlarini to'ldiring va 'Bashorat Qilish' tugmasini bosing")
    
    # Qo'shimcha ma'lumotlar
    with st.expander("ℹ️ Tizim haqida ma'lumot"):
        st.markdown("""
        ### Model haqida:
        - **Model turi:** Random Forest Classifier
        - **O'qitilgan ma'lumotlar:** 7000+ mijoz
        - **Aniqlik darajasi:** ~85%
        
        ### Ishlatilgan xususiyatlar:
        - Demografik ma'lumotlar (yosh, jins)
        - Xizmat muddati (tenure)
        - Xizmat turlari (telefon, internet)
        - Shartnoma va to'lov ma'lumotlari
        
        ### Churn bashorati:
        - **0.7+** - Yuqori xavf (qizil)
        - **0.3-0.7** - O'rtacha xavf (sariq)
        - **0.3-** - Past xavf (yashil)
        """)

if __name__ == "__main__":
    main()