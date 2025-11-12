'''
checkpoint.py — это системный модуль, который:

умеет сохранять состояние модели, чтобы не терять прогресс;

умеет загружать последний checkpoint при рестарте.

🔹 Минимальные задачи:
Задача	                                        Что делает
save_checkpoint(model, optimizer, epoch, path)	сохраняет веса и текущий номер эпохи
load_checkpoint(model, optimizer, path, device)	восстанавливает обучение
(опционально) get_latest_checkpoint(path)	       ищет последний чекпойнт

🔹 Минимальная реализация:
import torch
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)
    print(f"[CHECKPOINT] Saved: {path}")

def load_checkpoint(model, optimizer, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    print(f"[CHECKPOINT] Loaded from {path} (epoch {epoch})")
    return epoch


Использование:

# Сохранение
save_checkpoint(model, optimizer, epoch, "results/checkpoints/model_epoch_5.pth")

# Загрузка
start_epoch = load_checkpoint(model, optimizer, "results/checkpoints/model_epoch_5.pth", device)


💡 Даже если ты не будешь использовать их активно — само наличие checkpoint.py в проекте показывает понимание best practices (и это сильный сигнал для работодателя).
'''

# Цель: Сохранение и загрузка весов модели, контроль точек восстановления, возможность продолжить обучение.

# Верхнеуровневое содержимое: функции save_checkpoint, load_checkpoint, возможно обёртка для автосохранения по эпохам.


##


# В checkpoint_utils.py реализуй save_checkpoint(model, optimizer, epoch, path), load_checkpoint(path).

'''
logger.info(f"Checkpoint saved at epoch {epoch}")
'''