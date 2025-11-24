import matplotlib.pyplot as plt

def visualize_margins_simple(margins):
    plt.plot(margins, marker='o')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.ylabel("Отступ")
    plt.xlabel("Индекс объекта")
    plt.title("Визуализация отступов")
    plt.show()
