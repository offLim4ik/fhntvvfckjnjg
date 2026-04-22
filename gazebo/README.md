
------
Команды для настройки газебо в терминале виртуалки + доп коды по open cv
https://github.com/Galina-Basargina/aruco_flights
------
Генерация qr:
https://github.com/mikemka/gazebo_qr_gen
------
# 🚁 YOLO + Clover + Gazebo (Person Detection)

Простой гайд по запуску **распознавания людей (YOLO)** в симуляции **Clover (ROS + Gazebo)**.

---

## 📦 Требования

* Ubuntu + ROS Noetic
* Clover (simulator установлен)
* Python 3.8
* Интернет (для скачивания модели)

---

## 🚀 1. Запуск симулятора

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch clover_simulation simulator.launch
```

---

## 📷 2. Проверка камеры

```bash
rostopic list | grep main_camera
```

Ожидаемый результат:

```
/main_camera/image_raw
/main_camera/image_raw_throttled
```

Просмотр камеры:

```bash
rosrun image_view image_view image:=/main_camera/image_raw_throttled
```

---

## 🧠 3. Установка YOLO (под Python 3.8)

### Очистка (если были ошибки)

```bash
python3 -m pip uninstall -y torch torchvision torchaudio ultralytics typing-extensions
rm -rf ~/.cache/pip
```

### Установка зависимостей

```bash
python3 -m pip install --no-cache-dir "typing-extensions<4.13"
python3 -m pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --no-cache-dir ultralytics
```

---

## ✅ 4. Проверка установки

```bash
python3 -c "from ultralytics import YOLO; print('OK')"
```

---

## ⚙️ 5. Проверка потока камеры

```bash
rostopic hz /main_camera/image_raw_throttled
```

---

## 👁️ 6. Просмотр результата YOLO

```bash
rosrun image_view image_view image:=/yolo_debug
```

---

## 📁 7. Модель

Используется лёгкая модель:

```
yolo11n.pt
```

📌 При первом запуске скачивается автоматически.

---

## ⚠️ Важно

* YOLO ищет **людей (class = person)**
* Если в Gazebo нет человека → ничего не найдёт
* Минимальный размер объекта:

  ```
  ~150–200 пикселей в кадре
  ```

---

## 🔥 Типовые проблемы

### ❌ ModuleNotFoundError: ultralytics

👉 пакет не установлен → см. шаг 3

---

### ❌ No space left on device

👉 закончилась память:

```bash
df -h
```

---

### ❌ YOLO ничего не находит

Причины:

* в кадре нет человека
* объект слишком маленький
* плохое освещение

---

## 💡 Рекомендации

* Используй:

  ```
  /main_camera/image_raw_throttled
  ```

  (меньше нагрузка)

* Сначала тестируй **без полёта**

* Потом добавляй навигацию

---

## 🧩 Pipeline

```
Gazebo → Camera → ROS topic → YOLO → /yolo_debug → image_view
```

---

## 🏁 Итог

После выполнения:

* камера работает
* YOLO установлена
* человек детектится
* боксы отображаются

---

## 🚀 Next step

* центрирование по человеку
* автополёт
* интеграция с navigate

---
