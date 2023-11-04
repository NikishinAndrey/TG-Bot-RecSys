import telebot
from telebot import types
import sqlite3
import pandas as pd
import numpy as np
from implicit.nearest_neighbours import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import hstack, csr_matrix, lil_matrix
from tqdm.auto import tqdm
from random import random
import datetime
from Bot_token import BOT_TOKEN

bot = telebot.TeleBot(BOT_TOKEN)
model = ItemItemRecommender(K=100)

users = {}
grades = {}

epsilon = 0.15

ages = ["Меньше 18", "18-24", "24-35", "36-44", "45-54", "Более 54"]
genders = ["Мужчина", "Женщина"]
incomes = ["0-20", "20-40", "40-60", "60-90", "90-150", "150+"]
kids = ["Да", "Нет"]
keshback_like = ["👍", "👎", "СТОП"]

user_encoder = LabelEncoder()

#mutex
def pass_user(user_id, message):
    if user_id not in users:
        users[user_id] = {'flag': False, 'name': message.from_user.first_name}
        return True
    if users[user_id]['flag']:
        users[user_id]['flag'] = False
        return True
    return False

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Привет!")
    markup.add(btn1)

    msg = bot.send_message(message.chat.id, text=f"Привет, {message.from_user.first_name}! Позволь задать тебе несколько вопросов.", reply_markup=markup)
    bot.register_next_step_handler(msg, save_name)


@bot.message_handler(func=lambda msg: True)
def save_name(message):
    user_id = message.from_user.id
    if pass_user(user_id, message):
        users[user_id]['name'] = message.from_user.first_name

        conn = sqlite3.connect('../data/recsys_db.db')
        flag = pd.read_sql_query(f'SELECT COUNT(*) FROM Users WHERE id_user = {user_id}', conn)['COUNT(*)'][0] > 0
        conn.commit()
        conn.close()

        if flag:
            ask_grade(message)
        else:
            ask_age(message)


def ask_age(message):
    user_id = message.from_user.id
    users[user_id]['flag'] = True

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1, btn2, btn3, btn4, btn5, btn6 = (types.KeyboardButton(age) for age in ages)
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6)

    msg = bot.send_message(message.chat.id, text=f"Выберите ваш возраст", reply_markup=markup)
    bot.register_next_step_handler(msg, save_age)


@bot.message_handler(func=lambda msg: True)
def save_age(message):
    user_id = message.from_user.id
    if pass_user(user_id, message):
        if message.text in ages:
            users[user_id]['age'] = message.text
            ask_gender(message)
        else:
            ask_age(message)


def ask_gender(message):
    user_id = message.from_user.id
    users[user_id]['flag'] = True

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1, btn2 = (types.KeyboardButton(gender) for gender in genders)
    markup.add(btn1, btn2)

    msg = bot.send_message(message.chat.id, text=f"Выберите ваш пол", reply_markup=markup)
    bot.register_next_step_handler(msg, save_gender)


@bot.message_handler(func=lambda msg: True)
def save_gender(message):
    user_id = message.from_user.id
    if pass_user(user_id, message):
        if message.text in genders:
            users[user_id]['gender'] = message.text
            ask_income(message)
        else:
            ask_gender(message)


def ask_income(message):
    user_id = message.from_user.id
    users[user_id]['flag'] = True

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1, btn2, btn3, btn4, btn5, btn6 = (types.KeyboardButton(income) for income in incomes)
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6)

    msg = bot.send_message(message.chat.id, text=f"Выберите ваш доход (тыс. руб / мес)", reply_markup=markup)
    bot.register_next_step_handler(msg, save_income)


@bot.message_handler(func=lambda msg: True)
def save_income(message):
    user_id = message.from_user.id
    if pass_user(user_id, message):
        if message.text in incomes:
            users[user_id]['income'] = message.text
            ask_kids(message)
        else:
            ask_income(message)


def ask_kids(message):
    user_id = message.from_user.id
    users[user_id]['flag'] = True

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1, btn2 = (types.KeyboardButton(kid) for kid in kids)
    markup.add(btn1, btn2)

    msg = bot.send_message(message.chat.id, text=f"У вас есть дети?", reply_markup=markup)
    bot.register_next_step_handler(msg, save_user)


@bot.message_handler(func=lambda msg: True)
def save_user(message):
    user_id = message.from_user.id
    if pass_user(user_id, message):
        if message.text in kids:
            users[user_id]['kids'] = message.text

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add()

            user_data = (
                user_id,
                users[user_id]['name'],
                users[user_id]['age'],
                users[user_id]['gender'],
                users[user_id]['income'],
                users[user_id]['kids']
            )

            conn = sqlite3.connect('../data/recsys_db.db')
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO Users (id_user, name, age, gender, income, kids) VALUES (?, ?, ?, ?, ?, ?)', user_data)
            conn.commit()
            conn.close()
            bot.send_photo(message.chat.id, photo=open('images/default.png', 'rb'),
                          caption="Инфа записана. Спасибо!",
                          reply_markup=types.ReplyKeyboardRemove())
            ask_grade(message)
        else:
            ask_kids(message)


# Возвращает OHE csr матрицу, не выделяя nan, как отдельный признак
def onehot_without_nan(df):
    enc = OneHotEncoder()

    df_encoded = pd.DataFrame.sparse.from_spmatrix(
        enc.fit_transform(df.fillna('N/A')),
        columns=enc.get_feature_names_out()
    )

    mask = df_encoded.columns.str.contains(r"_N/A$")
    return csr_matrix(df_encoded.loc[:, ~mask])


def get_R_matrix(users_df, grades_df, num_items):
    user_encoder.fit(users_df['id_user'])

    # OHE-категориальные признаки юзеров
    users_enc = onehot_without_nan(users_df.drop(columns=['id_user',	'name']))
    users_ids = user_encoder.transform(users_df['id_user'])
    enc_matrix = lil_matrix((len(users_ids), users_enc.shape[1]))
    enc_matrix[pd.Index(users_ids)] = users_enc

    # Признаки айтемов юзеров (Можно также добавить продолжительность, время и признаки айтемов)
    matrix_shape = len(user_encoder.classes_), num_items
    users_ids = user_encoder.transform(grades_df['id_user'])
    sparse = csr_matrix((grades_df['grade'], (users_ids, grades_df['id_item'])), shape=matrix_shape)

    # Объединение всех признаков в матрицу
    return hstack([sparse, enc_matrix], format='csr')


def ask_grade(message):
    user_id = message.from_user.id
    # Используем Pandas для чтения данных из БД
    conn = sqlite3.connect('../data/recsys_db.db')
    users_df = pd.read_sql_query('SELECT * FROM Users', conn)
    grades_df = pd.read_sql_query('SELECT * FROM Grades', conn)
    num_items = pd.read_sql_query('SELECT COUNT(*) FROM Items', conn)['COUNT(*)'][0] + 1

    # Рекомендация с использованием ML (С добавлением epsilon-greedy strategy)
    if (users_df.shape[0] > 15) and (random() > epsilon):
        R = get_R_matrix(users_df, grades_df, num_items)
        model.fit(R)
        filter_items = np.arange(num_items, R.shape[1])

        user_idx = user_encoder.transform([user_id])[0]
        recommended_items = model.recommend(user_idx, user_items=R[user_idx], N=1, filter_items=filter_items, filter_already_liked_items=True)
        if recommended_items[1][0] != 0:
            # Если среди рекомендаций есть непросмотренные айтемы, то показать их
            item_idx = recommended_items[0][0]
            print(f"user: {user_id}. recommended item: {item_idx}.")
        else:
            # Показать случайный непросмотренный в противном случае
            zeros = np.argwhere(R.toarray()[user_idx] == 0)[:, 0]
            zeros = zeros[zeros < num_items - 1]

            if zeros.shape[0] > 0:
                item_idx = np.random.choice(zeros)
                print(f"user: {user_id}. unseen item: {item_idx}.")
            # ОБРАБОТКА СЛУЧАЯ, ЕСЛИ ВСЕ АЙТЕМЫ ПРОЛАЙКАНЫ
            else:
                print(f"user: {user_id}. all seen.")
                bot.send_photo(message.chat.id,
                               photo=open('images/default.png', 'rb'),
                               caption="Все кэшбеки просмотрены!",
                               reply_markup=types.ReplyKeyboardRemove())
                return
        item = pd.read_sql_query(f'SELECT * FROM Items WHERE id_item = {item_idx}', conn)
    # Случайная рекомендация
    else:
        item = pd.read_sql_query('SELECT * FROM Items ORDER BY RANDOM() LIMIT 1', conn)
        print(f"user: {user_id}. random item: {item['id_item'][0]}.")

    #num seeing items
    num_seeing_items = pd.read_sql_query(f'SELECT COUNT(*) FROM Grades WHERE id_user={user_id}', conn)['COUNT(*)'][0]

    conn.commit()
    conn.close()

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1, btn2, btn3 = (types.KeyboardButton(kesh_like) for kesh_like in keshback_like)
    markup.add(btn1, btn2, btn3)

    if user_id in grades:
        grades[user_id]['item_id'] = item['id_item'][0]
    else:
        grades[user_id] = {'item_id': item['id_item'][0]}

    msg = bot.send_photo(message.chat.id,
                         photo=open(f"images/{item['id_item'][0]}.png", 'rb'),
                         caption=f"{item['name'][0]}\n{item['cashback'][0]}\nПросмотрено: {num_seeing_items}",
                         reply_markup=markup)
    users[user_id]['flag'] = True
    bot.register_next_step_handler(msg, save_grade)


@bot.message_handler(func=lambda msg: True)
def save_grade(message):
    user_id = message.from_user.id
    if pass_user(user_id, message):
        if message.text in ["👍", "👎"]:
            grade = +1 if message.text == "👍" else -1
            item_id = int(grades[user_id]['item_id'])
            current_time = datetime.datetime.now()
            date_time = str(current_time.strftime("%m/%d/%Y %H:%M:%S"))
            item_data = (grade, user_id, item_id)

            conn = sqlite3.connect('../data/recsys_db.db')
            cursor = conn.cursor()

            cursor.execute('UPDATE Grades SET grade = ? WHERE id_user = ? AND id_item = ?', item_data)
            cursor.execute(f"UPDATE Grades SET time_grade = '{date_time}' WHERE id_user = {user_id} AND id_item = {item_id}")
            if cursor.rowcount == 0:
                cursor.execute('INSERT INTO Grades (grade, id_user, id_item) VALUES (?, ?, ?)', item_data)
                cursor.execute(f"UPDATE Grades SET time_grade = '{date_time}' WHERE id_user = {user_id} AND id_item = {item_id}")

            conn.commit()
            conn.close()

            ask_grade(message)
        elif message.text in ["СТОП"]:
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(types.KeyboardButton('Продолжить'))

            bot.send_photo(message.chat.id,
                          photo=open('images/default.png', 'rb'),
                          caption="Спасибо!",
                          reply_markup=markup)
            bot.register_next_step_handler(message, ask_grade)
        else:
            ask_grade(message)

bot.infinity_polling()
