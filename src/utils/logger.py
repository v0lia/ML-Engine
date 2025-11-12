'''
Основные цели:

показать структуру проекта уровня “production-ready” (наличие логирования);

иметь читаемые сообщения при запуске и ошибках.


🔹 Минимальные задачи:
Что логировать	        Пример
Запуск скриптов	        [INFO] Starting training..., [INFO] Starting evaluation...
Конфигурацию	        [INFO] Loaded config from src/config/config.yaml
Девайс и параметры	    [INFO] Using device: cuda (NVIDIA RTX 4070)
Размеры данных	        [INFO] Training set: 50000 samples, Validation: 10000 samples
Начало/конец эпох	    [INFO] Epoch 5/10 completed. Loss=0.34, Acc=89.2%
Сохранение чекпойнтов	[INFO] Saved checkpoint: results/checkpoints/model_epoch_5.pth
Исключения/ошибки	    [ERROR] Failed to load model weights: File not found


🔹 Минимальная реализация:
import logging

def get_logger(name="train", log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")

    # Консоль
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # (опционально) файл
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


Использование:

logger = get_logger("train")
logger.info("Starting training...")
logger.warning("Validation accuracy decreased!")
logger.error("Failed to load dataset.")


💡 На этом можно остановиться — этого достаточно, чтобы проект выглядел «живым» и структурированным.
'''
# seed.

# (Опционально): время каждой эпохи, GPU usage.

# Формат обычно простой: [TIME] [LEVEL] msg, пишется и в консоль, и в файл.