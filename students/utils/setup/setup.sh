#!/bin/bash

STUDENTS_ROOT="$(cd $(dirname ${BASH_SOURCE[0]})/../.. && pwd)"

# Функция для проверки корректности ввода kebab-case
validate_kebab_case() {
    if [[ ! "$1" =~ ^[a-z]+(\-[a-z]{2})?$ ]]; then
        echo "Ошибка: неверный формат. Используйте kebab-case (пример: ivanov-ii)" >&2
        return 1
    fi
}

# Функция для создания структуры лабораторной работы
create_lab_structure() {
    local student_name=$1
    local lab_num=$2
    
    lab_dir="$STUDENTS_ROOT/$student_name/lab$lab_num"
    
    # Если директория уже существует - пропускаем
    if [ -d "$lab_dir" ]; then
        echo "Директория lab$lab_num уже существует - пропускаем"
        return 1
    fi
    
    # Создаем структуру
    mkdir -p "$lab_dir/source"
    touch "$lab_dir/source/__init__.py"
    
    # Создаем README для лабораторной
    echo "# Лабораторная работа №$lab_num" > "$lab_dir/README.md"
    
    echo "Создана структура для lab$lab_num"
    return 0
}

# Функция для создания всех лабораторных работ (1-5)
create_all_labs() {
    local student_name=$1
    
    for lab_num in {1..5}; do
        create_lab_structure "$student_name" "$lab_num"
    done
}

# Запрос ввода пользователя
read -rp "Введите фамилию и инициалы в формате kebab-case (пример: ivanov-ii): " student_name

# Проверка формата ввода
validate_kebab_case "$student_name" || exit 1

# Создание корневой директории
student_dir="$STUDENTS_ROOT/$student_name"
if [ ! -d "$student_dir" ]; then
    mkdir -p "$student_dir"
    echo "Создана директория: $student_name"
fi

# Создание корневого README
root_readme="$STUDENTS_ROOT/$student_name/README.md"
if [ ! -f "$root_readme" ]; then
    echo "# Лабораторные работы по курсу \"Алгоритмы машинного обучения\"" > "$root_readme"
    echo "Создан корневой README"
fi

# Создание структуры для лабораторных работ
while true; do
    read -rp "Введите номер лабораторной работы для создания (0 для выхода, all для всех): " lab_input
    
    # Проверка на выход
    if [ "$lab_input" -eq 0 ] 2>/dev/null; then
        echo "Завершение работы скрипта."
        break
    fi

    # Проверка на ключевое слово "all"
    if [ "$lab_input" = "all" ]; then
        echo "Создание всех лабораторных работ (1-5)..."
        create_all_labs "$student_name"
        echo "Все лабораторные работы созданы."
        break
    fi

    # Проверка корректности номера
    if ! [[ "$lab_input" =~ ^[0-9]+$ ]]; then
        echo "Ошибка: номер должен быть корректным числом или all" >&2
        continue
    fi

    create_lab_structure "$student_name" "$lab_input"
    break
done
