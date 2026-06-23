def format_fn(tick_val, tick_pos):
    # если значение меньше 0.01, то переводим его в экспоненциальную форму
    if abs(tick_val) < 0.01 and tick_val != 0:
        # форматируем, заменяем экспоненту на красивую степень 10
        s = f"{tick_val:.1e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
        # превращаем 1.2e3 в $1.2 \cdot 10^{3}$
        base, exponent = s.split('e')
        return f"${base} \\cdot 10^{{{exponent}}}$"
    # если от 1 до 0.01, округляем до второго знака
    elif abs(tick_val) >= 0.01 and abs(tick_val) <1: 
        return f'{tick_val:.2f}'
    # если от 1 до 10, округляем до первого знаа
    elif abs(tick_val) >= 1 and abs(tick_val) <10:
        return f'{tick_val:.1f}'
    # если от 10 и выше, превращаем в целочисленное
    else: 
        return int(tick_val)