import cv2
import numpy as np
import os


# --- ЗАГРУЗКА И СОХРАНЕНИЕ ---

def load_source(path):
    """Чтение исходного изображения с диска."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {path}")
    print(f"Загружено: {path}, размер {image.shape}")
    return image


def save_result(path, image):
    """Запись изображения с проверкой результата."""
    success = cv2.imwrite(path, image)
    if success:
        print(f"OK -> {os.path.abspath(path)}")
    else:
        print(f"ОШИБКА: не удалось сохранить {path}")


# --- АФФИННЫЕ ПРЕОБРАЗОВАНИЯ ---

def scale_forward(image, kx, ky):
    """Прямое масштабирование."""
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * kx), int(h * ky)), interpolation=cv2.INTER_LINEAR)


def scale_inverse(image, kx, ky):
    """Обратное масштабирование."""
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w / kx), int(h / ky)), interpolation=cv2.INTER_AREA)


def shear_forward(image, factor):
    """Прямой скос."""
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (cols + int(rows * abs(factor)), rows))


def shear_inverse(image, factor):
    """Обратный скос."""
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, -factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (cols, rows))


# --- ФУНКЦИОНАЛЬНОЕ ПРЕОБРАЗОВАНИЕ ---

def functional_transform(image):
    """
    Нелинейное преобразование i = x^(3/5), j = y.
    Обратная функция: x = i^(5/3) — строго по методичке.
    """
    h, w = image.shape[:2]
    result = np.full_like(image, 255)

    for i in range(w):
        for j in range(h):
            x_src = int(i ** (5.0 / 3.0))
            if 0 <= x_src < w:
                result[j, i] = image[j, x_src]

    return result


# --- ОСНОВНОЙ БЛОК ---

if __name__ == "__main__":
    print("--- СТАРТ ---")

    SCALE_K = 1.5
    SHEAR_K = 0.5

    # 1. Загрузка
    img_orig = load_source("girl.png")

    # 2. Масштабирование (прямое и обратное)
    scaled = scale_forward(img_orig, SCALE_K, SCALE_K)
    save_result("scale.png", scaled)
    save_result("scale_back.png", scale_inverse(scaled, SCALE_K, SCALE_K))

    # 3. Скос (прямой и обратный)
    sheared = shear_forward(img_orig, SHEAR_K)
    save_result("shear.png", sheared)
    save_result("shear_back.png", shear_inverse(sheared, SHEAR_K))

    # 4. Функциональное преобразование
    save_result("functional.png", functional_transform(img_orig))

    print("--- ОБРАБОТКА ЗАВЕРШЕНА ---")