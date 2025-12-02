import matplotlib.pyplot as plt


def vis_LOO_errors(loo_errors, step=5):
    k_values = range(1, len(loo_errors) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, loo_errors, marker='o', linestyle='-')
    plt.title("Эмпирический риск (LOO ошибка) для разных k")
    plt.xlabel("k")
    plt.ylabel("LOO ошибка")
    
    # Подписи на оси X через step
    plt.xticks(k_values[::step])
    
    plt.grid(True)
    plt.show()
