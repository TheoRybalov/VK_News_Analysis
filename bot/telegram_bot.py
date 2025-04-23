import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pandas as pd
from datetime import datetime, timedelta
from main.news_analysis import analyze_news_sentiment
import os
import asyncio

CSV_PATH = "news_storage.csv"


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def format_news(df):
    messages = []
    for _, row in df.iterrows():
        pub = row["published"]
        sentiment = row["sentiment_prediction"]

        # Выбор эмодзи по тональности
        if sentiment == "позитивная":
            emoji = "🟢"
        elif sentiment == "негативная":
            emoji = "🔴"
        elif sentiment == "нейтральная":
            emoji = "⚪"
        else:
            emoji = "❓"

        # Форматируем дату
        date_str = pub.strftime('%Y-%m-%d %H:%M') if pd.notnull(pub) else 'дата неизвестна'

        # Собираем сообщение
        message = (
            f"*{row['title']}*\n\n"
            f"{emoji} {sentiment.upper()}\n\n\n"
            f"🏷️ Источник: {row['source']}\n"
            f"📅 {date_str}\n"
            f"🔗 [ссылка на новость]({row['link']})"
        )
        messages.append(message)
    return messages


def sentiment_summary(df):
    counts = df["sentiment_prediction"].value_counts()
    summary = "\n".join(f"{sent}: {counts.get(sent, 0)}" for sent in ["позитивная", "нейтральная", "негативная"])
    return f"📊 Новостной фон:\n{summary}"





async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот для анализа новостей по тональности.\n"
                                    "Напиши /analyze чтобы начать или /help для списка команд.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 Доступные команды:\n"
        "/analyze — показать 10 свежих новостей с тональностью\n"
        "/today — новости и фон за сегодня\n"
        "/week — новости и фон за последние 7 дней\n"
        "/last — последние 5 новостей\n"
        "/help — справка по командам"
    )


async def scheduled_news_analysis(context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = context.job.chat_id
        await context.bot.send_message(chat_id=chat_id, text="⏳ Начинаю периодический анализ...")

        # Асинхронный запуск анализа
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, analyze_news_sentiment)

        # Загружаем отправленные ранее
        if os.path.exists(CSV_PATH):
            sent_df = pd.read_csv(CSV_PATH)
            sent_df["published"] = pd.to_datetime(sent_df["published"], errors='coerce')
        else:
            sent_df = pd.DataFrame(columns=["link", "published"])

        # Преобразуем типы
        df["published"] = pd.to_datetime(df["published"], errors='coerce')
        df["link"] = df["link"].astype(str)
        sent_df["link"] = sent_df["link"].astype(str)

        # Выбираем 5 самых свежих, но неотправленных
        latest_df = df.head(5).sort_values(by='published', ascending=True)
        mask = ~latest_df.set_index(["link", "published"]).index.isin(
            sent_df.set_index(["link", "published"]).index
        )
        new_df = latest_df[mask]

        if new_df.empty:
            await context.bot.send_message(chat_id=chat_id, text="ℹ️ Новых новостей нет.")
            return

        messages = format_news(new_df)

        for msg in messages:
            await context.bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode='Markdown'
            )

        # Обновляем CSV отправленных
        updated_sent = pd.concat([sent_df, new_df[["link", "published"]]], ignore_index=True)
        updated_sent.drop_duplicates(inplace=True)
        updated_sent.to_csv(CSV_PATH, index=False)

    except Exception as e:
        logging.error(f"Scheduled job error: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка при автоматическом анализе: {e}")



async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = update.effective_chat.id
        job_queue = context.application.job_queue
        
        # Проверка существующих заданий
        current_jobs = job_queue.get_jobs_by_name(str(chat_id))
        
        if not current_jobs:
            job_queue.run_repeating(
                scheduled_news_analysis,
                interval=300,  # 5 минут
                first=10,      # Первый запуск через 10 сек
                chat_id=chat_id,
                name=str(chat_id)
            )
            msg = "✅ Автообновление активировано! Новости будут приходить каждые 5 минут"
        else:
            msg = "ℹ️ Автообновление уже активно"
        

        await update.message.reply_text("Анализирую новости...")

            
        await update.message.reply_text(msg)
        
    except Exception as e:
        await update.message.reply_text(f"🚨 Ошибка: {e}")
        logging.error(f"Analyze error: {e}")



async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyze_news_sentiment()
    today_df = df[df["published"].dt.date == datetime.now().date()]
    await update.message.reply_text(sentiment_summary(today_df))
    for msg in format_news(today_df.head(15)):
        await update.message.reply_text(msg)


async def week(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyze_news_sentiment()
    week_ago = datetime.now() - timedelta(days=7)
    week_df = df[df["published"] >= week_ago]
    await update.message.reply_text(sentiment_summary(week_df))
    for msg in format_news(week_df.head(20)):
        await update.message.reply_text(msg)


async def last(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyze_news_sentiment()
    for msg in format_news(df.head(5)):
        await update.message.reply_text(msg)



def run_bot():
    application = ApplicationBuilder().token("8013556995:AAHNXL5_8nxG_548hEhiByITWAkQgR5J8fY").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze))
    application.add_handler(CommandHandler("today", today))
    application.add_handler(CommandHandler("week", week))
    application.add_handler(CommandHandler("last", last))

    print("Бот запущен")
    application.run_polling()
