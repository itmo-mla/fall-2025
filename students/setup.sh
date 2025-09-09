#!/bin/bash

# Функция для проверки корректности ввода kebab-case
validate_kebab_case() {
    if [[ ! "$1" =~ ^[a-z]+(\-[a-z]{2})?$ ]]; then
        echo "Ошибка: неверный формат. Используйте kebab-case (пример: ivanov-ii)" >&2
        return 1
    fi
}

# Запрос ввода пользователя
read -rp "Введите фамилию и инициалы в формате kebab-case (пример: ivanov-ii): " student_name

# Проверка формата ввода
validate_kebab_case "$student_name" || exit 1

# Создание корневой директории
if [ ! -d "$student_name" ]; then
    mkdir -p "$student_name"
    echo "Создана директория: $student_name"
fi

# Создание корневого README
root_readme="$student_name/README.md"
if [ ! -f "$root_readme" ]; then
    echo "# Лабораторные работы по курсу \"Алгоритмы машинного обучения\"" > "$root_readme"
    echo "Создан корневой README"
fi

# Создание структуры для лабораторных работ
for lab_num in {1..5}; do
    lab_dir="$student_name/lab$lab_num"
    
    # Если директория уже существует - пропускаем
    if [ -d "$lab_dir" ]; then
        echo "Директория lab$lab_num уже существует - пропускаем"
        continue
    fi
    
    # Создаем структуру
    mkdir -p "$lab_dir/source"
    touch "$lab_dir/source/__init__.py"
    
    # Создаем README для лабораторной
    echo "# Лабораторная работа №$lab_num" > "$lab_dir/README.md"
    
    echo "Создана структура для lab$lab_num"
done

echo "Готово! Структура создана для студента: $student_name"
