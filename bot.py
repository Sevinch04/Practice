import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import pandas as pd
import joblib
import os

# Logging sozlash
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
COLLECTING_DATA = 1

# Global o'zgaruvchilar
model = None
label_encoders = None
feature_names = None

def load_models():
    """Modellarni yuklash"""
    global model, label_encoders, feature_names
    
    try:
        # Models papkasiga yo'l
        model_dir = '../models'  # yoki to'liq yo'l
        
        # Agar models papkasi hozirgi papkada bo'lsa
        if not os.path.exists(model_dir):
            model_dir = 'models'
        
        # Agar hali ham yo'q bo'lsa, ota papkada qidirish
        if not os.path.exists(model_dir):
            model_dir = '../models'
        
        # Modellarni yuklash
        model = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
        label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        
        logger.info("✅ Modellar muvaffaqiyatli yuklandi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Modellarni yuklashda xatolik: {e}")
        return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bot boshlanish"""
    user = update.effective_user
    
    welcome_message = f"""
🤖 Salom {user.first_name}!

📱 **Telekom Churn Bashorat Bot**ga xush kelibsiz!

Bu bot mijozning telekom kompaniyasini tark etish ehtimolini bashorat qiladi.

Buyruqlar:
• /start - Botni qayta boshlash
• /predict - Churn bashoratini boshlash
• /help - Yordam

Bashorat qilish uchun /predict ni yuboring.
    """
    
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Yordam buyrug'i"""
    help_text = """
🆘 **YORDAM**

**Asosiy buyruqlar:**
• /start - Botni qayta ishga tushirish
• /predict - Churn bashoratini boshlash
• /help - Bu yordam xabari

**Bot qanday ishlaydi:**
1. /predict buyrug'ini yuboring
2. So'ralgan formatda ma'lumot yuboring
3. Bot bashorat natijasini qaytaradi

**Ma'lumot formati:**
gender,senior,partner,dependents,tenure,phone,internet,contract,monthly,total

**Misol:**
Male,0,No,No,12,Yes,DSL,Month-to-month,65.0,780.0

**Tushuntirish:**
- gender: Male/Female
- senior: 0 (yo'q) yoki 1 (ha)
- partner: Yes/No
- dependents: Yes/No  
- tenure: xizmat muddati (oylar)
- phone: Yes/No
- internet: DSL/Fiber optic/No
- contract: Month-to-month/One year/Two year
- monthly: oylik to'lov ($)
- total: jami to'lov ($)
    """
    await update.message.reply_text(help_text)

async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Bashorat jarayonini boshlash"""
    if model is None:
        await update.message.reply_text(
            "❌ Model yuklanmagan. Bot administrator bilan bog'laning."
        )
        return ConversationHandler.END
    
    await update.message.reply_text(
        "📝 **Ma'lumotlarni quyidagi formatda yuboring:**\n\n"
        "`gender,senior,partner,dependents,tenure,phone,internet,contract,monthly,total`\n\n"
        "**Misol:**\n"
        "`Male,0,No,No,12,Yes,DSL,Month-to-month,65.0,780.0`\n\n"
        "**Yoki /cancel buyrug'i bilan bekor qiling**",
        parse_mode='Markdown'
    )
    
    return COLLECTING_DATA

async def process_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Ma'lumotlarni qayta ishlash va bashorat qilish"""
    
    if model is None:
        await update.message.reply_text("❌ Model mavjud emas!")
        return ConversationHandler.END
    
    try:
        # Ma'lumotlarni ajratish
        data_text = update.message.text.strip()
        data_parts = [part.strip() for part in data_text.split(',')]
        
        if len(data_parts) != 10:
            await update.message.reply_text(
                f"❌ 10 ta qiymat kerak, lekin {len(data_parts)} ta berildi!\n"
                "To'g'ri format: gender,senior,partner,dependents,tenure,phone,internet,contract,monthly,total"
            )
            return COLLECTING_DATA
        
        # Ma'lumotlarni dictionary ga aylantirish
        user_data = {
            'gender': data_parts[0],
            'SeniorCitizen': int(data_parts[1]),
            'Partner': data_parts[2],
            'Dependents': data_parts[3],
            'tenure': float(data_parts[4]),
            'PhoneService': data_parts[5],
            'MultipleLines': 'No',  # Soddalashgan
            'InternetService': data_parts[6],
            'OnlineSecurity': 'No',  # Soddalashgan
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': data_parts[7],
            'PaperlessBilling': 'Yes',  # Soddalashgan
            'PaymentMethod': 'Electronic check',  # Soddalashgan
            'MonthlyCharges': float(data_parts[8]),
            'TotalCharges': float(data_parts[9])
        }
        
        # Ma'lumotlarni kodlash
        input_df = pd.DataFrame([user_data])
        
        # Label encoders bilan kodlash
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except ValueError:
                    # Agar yangi qiymat bo'lsa, default qiymat berish
                    input_df[col] = 0
        
        # Feature names tartibida joylash
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        # Bashorat qilish
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Natijalarni tayyorlash
        churn_prob = probability[1] * 100
        stay_prob = probability[0] * 100
        
        if prediction == 1:
            result_emoji = "⚠️"
            result_text = "KETISHI MUMKIN"
            risk_level = "🔴 YUQORI XAVF"
        else:
            result_emoji = "✅" 
            result_text = "QOLADI"
            risk_level = "🟢 PAST XAVF"
        
        # Natija xabari
        result_message = f"""
{result_emoji} **BASHORAT NATIJASI** {result_emoji}

🎯 **Natija:** {result_text}
📊 **Xavf darajasi:** {risk_level}

📈 **Ehtimolliklar:**
• Qolish: {stay_prob:.1f}%
• Ketish: {churn_prob:.1f}%

📋 **Kiritilgan ma'lumotlar:**
• Jins: {data_parts[0]}
• Keksa fuqaro: {'Ha' if data_parts[1] == '1' else "Yo'q"}
• Sherik: {data_parts[2]}
• Xizmat muddati: {data_parts[4]} oy
• Shartnoma: {data_parts[7]}
• Oylik to'lov: ${data_parts[8]}

💡 **Tavsiyalar:**
"""
        
        if prediction == 1:
            result_message += """
• 🎯 Mijoz bilan shaxsiy aloqa o'rnating
• 💰 Maxsus chegirmalar taklif qiling  
• 📞 Qo'ng'iroq qiling va muammolarni aniqlang
• 🎁 Loyalty dasturiga taklif qiling
• 📄 Uzoq muddatli shartnoma taklif qiling
"""
        else:
            result_message += """
• 👏 Mijozni rag'batlantiring
• 📢 Yangi xizmatlar haqida xabar bering
• 🎁 Referral dasturini taklif qiling
• ⭐ Premium xizmatlarni ko'rsating
• 📝 Feedback so'rang
"""
        
        result_message += f"\n🔄 **Qayta bashorat:** /predict"
        
        await update.message.reply_text(result_message, parse_mode='Markdown')
        
        return ConversationHandler.END
        
    except ValueError as e:
        await update.message.reply_text(
            f"❌ Ma'lumot formati noto'g'ri: {e}\n\n"
            "Raqamlar to'g'ri formatda kiritilganini tekshiring.\n"
            "Misol: Male,0,No,No,12,Yes,DSL,Month-to-month,65.0,780.0"
        )
        return COLLECTING_DATA
        
    except Exception as e:
        logger.error(f"Bashorat xatoligi: {e}")
        await update.message.reply_text(
            f"❌ Bashorat qilishda xatolik yuz berdi.\n"
            f"Iltimos, ma'lumotlarni qayta tekshiring yoki /cancel ni bosing."
        )
        return COLLECTING_DATA

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Conversation ni bekor qilish"""
    await update.message.reply_text(
        "🚫 Bashorat jarayoni bekor qilindi.\n\n"
        "Qayta boshlash uchun: /predict",
    )
    return ConversationHandler.END

def main():
    """Asosiy funksiya"""
    print("🤖 Telegram bot ishga tushirilmoqda...")
    
    # Modellarni yuklash
    if not load_models():
        print("❌ Modellar yuklanmadi! Bot to'liq ishlamaydi.")
        print("Avval churn_analysis.ipynb ni ishga tushirib modellarni yarating.")
    
    # Bot token (o'zgartirishingiz kerak!)
    TOKEN = "7677818705:AAGxjfZAH0DnI99p9xUMpwrmZmq6ABbZd8s"
    
    if TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ DIQQAT: Bot token o'rnatilmagan!")
        print("1. @BotFather ga boring")
        print("2. /newbot buyrug'ini yuboring") 
        print("3. Token oling va YOUR_BOT_TOKEN_HERE ni almashtiring")
        return
    
    # Application yaratish
    application = Application.builder().token(TOKEN).build()
    
    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("predict", predict_start)],
        states={
            COLLECTING_DATA: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_prediction)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    # Handler'larni qo'shish
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(conv_handler)
    
    # Botni ishga tushirish
    print("✅ Bot muvaffaqiyatli ishga tushdi!")
    print("📱 Telegram'da bot bilan /start yuboring")
    print("⏹️ To'xtatish uchun Ctrl+C bosing")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot to'xtatildi")
    except Exception as e:
        print(f"❌ Kritik xatolik: {e}")